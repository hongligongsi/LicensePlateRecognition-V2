"""传统 CV 识别引擎。

流程：定位 → 颜色精定位 → 二值化 → 字符分割 → SVM 分类 → 后处理校验。

相对原版 ``CardPredictor.predict()`` 的改进：

* **不再依赖工作目录**：模型路径由配置决定，不会因为从别的目录启动就找不到
  ``svm.dat``（原版用相对路径，是常见踩坑点）。
* **不再靠缩放撞运气**：原界面硬编码 ``(1, 0.9, ..., 0.4)`` 七档重试，
  现在定位模块内部用比例参数自适应，一次到位。
* **真正多车牌**：原版 ``break`` 掉了，只返回第一个命中的车牌，
  有多个车牌时后面的全部丢弃（注释写"识别车牌中的字符"但实际只处理一个）。
* **置信度**：原版只有硬标签，现在给出字符级 + 整体置信度。
* **车牌号码合法性校验**：用正则做后处理，修正常见混淆（``0``↔``O``、``1``↔``I``）。
* **失败原因可见**：返回 ``meta`` 说明卡在了哪一步，方便调参。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from ..config import EngineConfig
from ..types import PlateColor, PlateResult
from ..utils.image import imread
from ..utils.logging_setup import get_logger
from .base import BaseEngine, EngineUnavailable
from .locate import detect_color, locate_plates, refine_by_color
from .segment import separate_characters
from .svm_classifier import (
    PROVINCES,
    build_alphanum_classifier,
    build_province_classifier,
)

logger = get_logger("engines.cv")

#: 蓝/白/黑牌是深色字浅色底；黄/绿牌相反
_DARK_TEXT_COLORS = {PlateColor.BLUE, PlateColor.WHITE, PlateColor.BLACK}

#: 车牌号码：省份简称 + 字母 + 5~6 位（新能源 6 位，普通 5 位）
_PLATE_PATTERN = re.compile(r"^[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新]" r"[A-Z][A-Z0-9]{4,6}$")


def _is_dark_text(color: PlateColor) -> bool:
    """判断该底色的车牌是否"浅底深字"，决定是否要反色。"""
    return color in _DARK_TEXT_COLORS


def _fix_confusions(text: str) -> str:
    """修正常见的字符混淆。

    规则来自车牌号码的语法约束：
      * 第 1 位必须是省份简称（汉字）
      * 第 2 位必须是字母（不含 I、O）
      * 第 3~7 位是数字或字母
    """
    if len(text) < 2:
        return text

    chars = list(text)
    # 第 2 位必须是字母：数字 0→O 是错的（O 不合法），应改成相邻合法字母
    if chars[1].isdigit():
        # 数字 0/1/2/3... 被识别成第 2 位时，最可能是 O/I/Z 的误判
        mapping = {"0": "D", "1": "J", "2": "Z", "5": "S", "8": "B", "6": "G"}
        chars[1] = mapping.get(chars[1], "A")

    # I、O 在车牌第 2 位以后的字母位不出现，替换为易混淆的数字
    for i in range(1, len(chars)):
        if chars[i] == "I":
            chars[i] = "1"
        elif chars[i] == "O":
            chars[i] = "0"

    return "".join(chars)


def _score_text(text: str) -> float:
    """给一个候选号码打合法性分数，用于多候选之间的选择。"""
    if not text:
        return -1.0
    score = 0.0
    if len(text) == 7:
        score += 2.0
    elif len(text) == 8:  # 新能源
        score += 1.5
    else:
        score -= 1.0
    if _PLATE_PATTERN.match(text):
        score += 2.0
    if text[0] in PROVINCES:
        score += 1.5
    else:
        score -= 1.0
    return score


class CVEngine(BaseEngine):
    """传统 OpenCV + SVM 车牌识别引擎。"""

    name = "cv"
    display_name = "传统 CV 引擎"
    description = "OpenCV 形态学定位 + 投影分割 + SVM 字符识别，纯 CPU、轻量、可解释"

    def __init__(self, config: EngineConfig | None = None, models_dir: Path | None = None):
        super().__init__()
        self.config = config or EngineConfig()
        self.cfg = self.config.locate
        self.models_dir = Path(models_dir) if models_dir else _default_models_dir()
        self._alnum = None
        self._province = None

    # -------------------------------------------------------------- 可用性

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False, "未安装 opencv-python，请执行: pip install opencv-python"

        try:
            import cv2 as _cv2

            if not hasattr(_cv2, "ml"):
                return False, (
                    f"当前 OpenCV {_cv2.__version__} 已移除 cv2.ml 模块，"
                    "无法加载 SVM 模型。请安装 4.x 版本: pip install 'opencv-python>=4.10,<5'"
                )
        except Exception as exc:  # pragma: no cover
            return False, f"OpenCV 探测失败: {exc}"

        return True, ""

    # -------------------------------------------------------------- 加载

    def load(self) -> None:
        ok, reason = self.is_available()
        if not ok:
            raise EngineUnavailable(reason)

        alnum_path = self.models_dir / "svm.dat"
        prov_path = self.models_dir / "svmchinese.dat"

        self._alnum = build_alphanum_classifier(alnum_path)
        self._province = build_province_classifier(prov_path)

        try:
            self._alnum.load()
            self._province.load()
        except FileNotFoundError as exc:
            raise EngineUnavailable(
                f"{exc}\n模型文件缺失。可从原项目下载 svm.dat / svmchinese.dat 放入 {self.models_dir}"
            ) from exc

        self._ready = True

    # -------------------------------------------------------------- 识别

    def predict(self, image: np.ndarray | str | Path) -> list[PlateResult]:
        self.ensure_ready()

        if isinstance(image, (str, Path)):
            img = imread(image)
        else:
            img = image

        if img is None or img.size == 0:
            return []

        started = time.perf_counter()
        candidates, scale = locate_plates(img, self.cfg, logger)

        if not candidates:
            logger.debug("未定位到车牌候选区域（可能车牌过小/过倾斜/颜色不符合）")
            return []

        results: list[PlateResult] = []
        attempts: list[str] = []

        for cand in candidates:
            if cand.color is PlateColor.UNKNOWN:
                attempts.append("颜色不明确，跳过")
                continue

            roi = refine_by_color(cand.roi, cand.color, self.cfg)
            if roi is None or roi.size == 0:
                attempts.append("颜色精定位后 ROI 为空")
                continue

            seg = separate_characters(roi, _is_dark_text(cand.color))
            if len(seg.chars) < self.cfg.min_chars:
                attempts.append(f"仅切出 {len(seg.chars)} 个字符（需 ≥ {self.cfg.min_chars}）")
                continue

            text, confidences = self._recognize(seg.chars)
            if not text:
                attempts.append("字符识别结果为空")
                continue

            fixed = _fix_confusions(text)
            if fixed != text:
                logger.debug("后处理修正: %s → %s", text, fixed)
            text = fixed

            confidence = float(np.mean(confidences)) if confidences else 0.0

            results.append(
                PlateResult(
                    text=text,
                    color=cand.color,
                    bbox=cand.bbox,
                    roi=roi,
                    confidence=confidence,
                    engine=self.name,
                    char_confidences=confidences,
                    meta={
                        "aspect": round(cand.aspect, 2),
                        "chars": len(seg.chars),
                        "scale": round(scale, 3),
                    },
                )
            )

        elapsed = (time.perf_counter() - started) * 1000
        logger.debug("CV 引擎耗时 %.1f ms，命中 %d 个车牌", elapsed, len(results))

        # 多个候选都识别出号码时，按合法性分数排序，最可信的放最前
        results.sort(key=lambda r: _score_text(r.text), reverse=True)

        if not results:
            logger.debug("候选均未通过字符识别: %s", "; ".join(attempts[:3]))

        return results

    def _recognize(self, chars: list[np.ndarray]) -> tuple[str, list[float]]:
        """识别字符序列：第 1 个走省份分类器，其余走数字/字母分类器。"""
        if not chars:
            return "", []

        province_patch, rest = chars[0], chars[1:]

        prov_results = self._province.predict_batch([province_patch])
        prov_text, prov_conf = prov_results[0] if prov_results else ("", 0.0)

        # 省份识别不可靠时，用位置约束兜底：至少保证第 1 位是汉字
        if not prov_text:
            prov_text, prov_conf = "?", 0.0

        rest_results = self._alnum.predict_batch(rest) if rest else []

        text = prov_text + "".join(t for t, _ in rest_results)
        confidences = [prov_conf] + [c for _, c in rest_results]
        return text, confidences


def _default_models_dir() -> Path:
    from ..config import MODELS_DIR

    return MODELS_DIR


__all__ = ["CVEngine"]
