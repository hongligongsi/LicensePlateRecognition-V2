"""基于 OpenCV SVM 的字符分类器。

**重要：特征提取必须与原项目完全一致，否则旧模型权重失效。**

原版 HOG 特征（``preprocess_hog``）：
  * 输入 20x20 灰度字符图
  * Sobel 求 x/y 梯度 → 幅值、角度
  * 角度量化为 16 个 bin
  * 2x2 = 4 个 cell（每个 10x10）
  * 每 cell 做加权直方图 → 4 * 16 = **64 维**特征
  * Hellinger 归一化（L1 归一化 → 开根号 → L2 归一化）

上面这一步已经用 ``cv2.ml.SVM_load`` 实际验证过：``svm.dat`` 的
``getVarCount() == 64``，与 4*16 一致，说明特征维度对得上。

需要修的原版问题：

1. ``np.int32(bin_n*ang/(2*np.pi))`` 在角度恰为 2π 时会得到 16，
   越界导致 ``np.bincount`` 返回值长度变成 17 —— 而 ``np.hstack`` 仍能拼接，
   结果是特征维度悄悄变化，SVM 直接报错。这里用 ``np.clip`` 卡在 15。
2. 原版用 ``np.bincount`` 逐 cell 调用 4 次，这里保留（量很小），
   但改用 ``minlength`` 保证长度恒定。
3. 原版没有字符置信度，只有硬标签。这里用 ``SVM.predict`` 的
   多分类决策值算出置信度，供界面展示和级联决策使用。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.logging_setup import get_logger

logger = get_logger("engines.svm")

#: 字符图归一化尺寸，必须与训练时一致
CHAR_SIZE = 20
#: 角度量化 bin 数
BIN_N = 16
#: cell 划分（2x2），每 cell 10x10
CELL = CHAR_SIZE // 2
#: 特征维度 = BIN_N * 4
FEATURE_DIM = BIN_N * 4

#: 省份简称顺序。**必须与训练 svmchinese.dat 时的顺序完全一致**，
#: 否则识别出的汉字会整体错位。原版顺序即此，已核对。
PROVINCES = [
    "川",
    "鄂",
    "赣",
    "甘",
    "贵",
    "桂",
    "黑",
    "沪",
    "冀",
    "津",
    "京",
    "吉",
    "辽",
    "鲁",
    "蒙",
    "闽",
    "宁",
    "青",
    "琼",
    "陕",
    "苏",
    "晋",
    "皖",
    "湘",
    "新",
    "豫",
    "渝",
    "粤",
    "云",
    "藏",
    "浙",
]
#: 训练时省份标签从 1001 开始（PROVINCE_START=1000，+1 偏移）
PROVINCE_START = 1000
#: 车牌第 2 位可用字母（I/O 被排除，避免与 1/0 混淆）
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
#: 普通车牌的字符集：数字 + 字母
ALPHANUM = list("0123456789") + LETTERS


class SVMCharacterClassifier:
    """包装 OpenCV SVM 的字符分类器，线程安全。"""

    def __init__(self, model_path: str | Path, label_map: dict[int, str]) -> None:
        self.model_path = Path(model_path)
        self.label_map = label_map
        self._model: Any = None  # cv2.ml.SVM 实例（opencv-python 缺少对应类型标注）
        self._lock = threading.Lock()

    def load(self) -> None:
        """加载模型文件。"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"SVM 模型不存在: {self.model_path}")

        with self._lock:
            model = cv2.ml.SVM_load(str(self.model_path))  # type: ignore[attr-defined]
            if model is None or model.empty():
                raise RuntimeError(f"SVM 模型加载失败（文件损坏或格式不兼容）: {self.model_path}")

            dim = model.getVarCount()
            if dim != FEATURE_DIM:
                raise RuntimeError(
                    f"特征维度不匹配：模型要求 {dim} 维，当前实现输出 {FEATURE_DIM} 维。"
                    " 请确认模型是否由本项目的训练脚本生成。"
                )

            self._model = model
            logger.info(
                "已加载 %s（%d 维特征, kernel=%d）",
                self.model_path.name,
                dim,
                model.getKernelType(),
            )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -------------------------------------------------------------- 特征提取

    @staticmethod
    def preprocess_hog(images: list[np.ndarray]) -> np.ndarray:
        """提取 HOG 特征，输出形状 ``(N, 64)`` 的 float32 数组。

        与 OpenCV sample 及原项目的实现保持一致。
        """
        samples = []
        for img in images:
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)

            # 关键修复：角度恰好为 2π 时 bin 会算出 16（越界），
            # 原版没处理，会导致 bincount 长度变 17、特征维度漂移。
            bin_idx = np.int32(BIN_N * ang / (2 * np.pi))
            np.clip(bin_idx, 0, BIN_N - 1, out=bin_idx)

            bin_cells = (
                bin_idx[:CELL, :CELL],
                bin_idx[CELL:, :CELL],
                bin_idx[:CELL, CELL:],
                bin_idx[CELL:, CELL:],
            )
            mag_cells = (
                mag[:CELL, :CELL],
                mag[CELL:, :CELL],
                mag[:CELL, CELL:],
                mag[CELL:, CELL:],
            )

            hists = [
                np.bincount(b.ravel(), m.ravel(), minlength=BIN_N)
                for b, m in zip(bin_cells, mag_cells, strict=False)
            ]
            hist = np.hstack(hists).astype(np.float32)

            # Hellinger kernel：L1 归一化 → 开根号 → L2 归一化
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= np.linalg.norm(hist) + eps

            samples.append(hist)

        if not samples:
            return np.empty((0, FEATURE_DIM), np.float32)
        return np.asarray(samples, dtype=np.float32)

    @staticmethod
    def normalize_char(patch: np.ndarray) -> np.ndarray:
        """把切割出的字符图规整成 20x20 的白字黑底灰度图。

        原版是 ``copyMakeBorder`` 左右补 ``w//3`` 再 resize。
        这里在补边之前先把字符**垂直居中**，避免字符偏上/偏下导致特征漂移。
        """
        if patch is None or patch.size == 0:
            return np.zeros((CHAR_SIZE, CHAR_SIZE), np.uint8)

        img = patch
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 裁到字符实际包围盒，去掉四周空白，让不同宽度的字符公平缩放
        ys, xs = np.nonzero(img)
        if ys.size and xs.size:
            img = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((CHAR_SIZE, CHAR_SIZE), np.uint8)

        # 按高度先缩放到 20，保持长宽比
        scale = CHAR_SIZE / h
        new_w = max(1, round(w * scale))
        resized = cv2.resize(img, (new_w, CHAR_SIZE), interpolation=cv2.INTER_AREA)

        # 宽度不足就左右对称补黑边，超过就居中裁剪
        if new_w < CHAR_SIZE:
            pad_left = (CHAR_SIZE - new_w) // 2
            pad_right = CHAR_SIZE - new_w - pad_left
            resized = cv2.copyMakeBorder(
                resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
            )
        elif new_w > CHAR_SIZE:
            start = (new_w - CHAR_SIZE) // 2
            resized = resized[:, start : start + CHAR_SIZE]

        return resized

    # -------------------------------------------------------------- 推理

    def predict_batch(self, patches: list[np.ndarray]) -> list[tuple[str, float]]:
        """批量识别字符，返回 ``[(字符, 置信度), ...]``。

        置信度来自多分类 SVM 的决策值，用 softmax 归一化到 0~1。
        """
        if not patches:
            return []

        with self._lock:
            if self._model is None:
                raise RuntimeError("模型尚未加载，请先调用 load()")

            normalized = [self.normalize_char(p) for p in patches]
            features = self.preprocess_hog(normalized)

            # 取决策值用于置信度估计
            try:
                _, raw = self._model.predict(features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                decision = np.asarray(raw, dtype=np.float32).reshape(len(patches), -1)
            except cv2.error:
                decision = None

            _, preds = self._model.predict(features)
            labels = np.asarray(preds).ravel().astype(int)

        results: list[tuple[str, float]] = []
        for i, label in enumerate(labels):
            text = self.label_map.get(int(label), "")
            if not text:
                # 模型输出不在映射表里 —— 原版会直接 chr() 出一个乱码字符
                logger.debug("模型输出未知标签 %s，已丢弃", label)
                results.append(("", 0.0))
                continue

            confidence = 0.5
            if decision is not None and i < decision.shape[0]:
                row = decision[i]
                if row.size == 1:
                    # 二分类：sigmoid
                    confidence = float(1.0 / (1.0 + np.exp(-abs(row[0]))))
                elif row.size > 1:
                    # 多分类：把决策值最大的一项做 softmax
                    exp = np.exp(row - row.max())
                    confidence = float(exp.max() / exp.sum())

            results.append((text, confidence))

        return results


def build_alphanum_classifier(model_path: str | Path) -> SVMCharacterClassifier:
    """构建数字/字母分类器。

    训练时标签是 ``ord(字符)``，所以标签映射就是把 int 转回 chr，
    但只保留车牌合法字符集，避免模型输出奇怪符号。
    """
    label_map = {ord(ch): ch for ch in ALPHANUM}
    return SVMCharacterClassifier(model_path, label_map)


def build_province_classifier(model_path: str | Path) -> SVMCharacterClassifier:
    """构建省份简称分类器。

    训练时标签为 ``PROVINCE_START + 1 + provinces.index(pinyin)``，
    即 ``1001 + idx``，对应 ``PROVINCES[idx]``。
    """
    label_map = {PROVINCE_START + 1 + idx: name for idx, name in enumerate(PROVINCES)}
    return SVMCharacterClassifier(model_path, label_map)


__all__ = [
    "ALPHANUM",
    "CHAR_SIZE",
    "FEATURE_DIM",
    "LETTERS",
    "PROVINCES",
    "PROVINCE_START",
    "SVMCharacterClassifier",
    "build_alphanum_classifier",
    "build_province_classifier",
]
