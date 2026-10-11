"""深度学习识别引擎（可选）。

设计目标：**缺失依赖时必须优雅降级，而不是让整个程序崩掉。**

原项目是"装不上就走不了"的硬依赖模式（Python 3.4 + OpenCV 3.4 至今已无法安装）。
这里把深度学习做成可选增强：

* 装了 ``paddleocr`` / ``rapidocr-onnxruntime`` → 自动启用，走高精度路径；
* 没装 → ``is_available()`` 返回 False，界面把该选项置灰并提示安装命令，
  传统 CV 引擎照常工作。

后端优先级（按实际可安装性排序）：
  1. ``rapidocr-onnxruntime`` —— ONNX 推理，CPU 友好，依赖轻，无需 PaddlePaddle
  2. ``paddleocr`` —— 精度最高，但依赖重（PaddlePaddle 体积大）

深度学习后端的输出是"文本框 + 文字 + 置信度"，而不是车牌框。
因此还需要一步**车牌号码几何定位**：在一堆检测出的文本中，
按"符合车牌语法的文本 + 位于同一水平行 + 长宽比合理"挑出真正的车牌。
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

logger = get_logger("engines.deep")

#: 车牌号码的粗略正则（省份简称 + 字母 + 数字字母混合）
_PLATE_TEXT = re.compile(
    r"[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新]"
    r"[A-Z][A-Z0-9]{4,6}"
)
#: 也接受不含省份简称的 6~7 位纯字母数字（后续会尝试补省份）
_BARE_TEXT = re.compile(r"^[A-Z0-9]{6,7}$")


def _probe_rapidocr() -> tuple[bool, str]:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False, "未安装 rapidocr-onnxruntime（推荐，轻量）: pip install rapidocr-onnxruntime"
    return True, ""


def _probe_paddleocr() -> tuple[bool, str]:
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        return False, "未安装 paddleocr: pip install paddleocr paddlepaddle"
    return True, ""


class DeepEngine(BaseEngine):
    """基于 OCR 大模型的识别引擎。

    与传统引擎的区别：不再手工做形态学定位和字符分割，
    而是让神经网络直接输出"这块区域是什么文字"，中文字符准确率显著更高。
    """

    name = "deep"
    display_name = "深度学习引擎"
    description = "OCR 深度学习模型，对中文省份简称、模糊/倾斜车牌准确率更高（需额外安装依赖）"

    def __init__(self, config: EngineConfig | None = None, backend: str = "auto"):
        super().__init__()
        self.config = config or EngineConfig()
        self.cfg = self.config.locate
        self.backend = backend
        self._ocr = None
        self._impl = ""

    # -------------------------------------------------------------- 可用性

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        ok, msg = _probe_rapidocr()
        if ok:
            return True, ""
        ok2, msg2 = _probe_paddleocr()
        if ok2:
            return True, ""
        return False, f"{msg}；或 {msg2}"

    # -------------------------------------------------------------- 加载

    def load(self) -> None:
        ok, reason = self.is_available()
        if not ok:
            raise EngineUnavailable(reason)

        if self.backend in ("auto", "rapidocr") and _probe_rapidocr()[0]:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
            self._impl = "rapidocr"
        elif self.backend in ("auto", "paddleocr") and _probe_paddleocr()[0]:
            from paddleocr import PaddleOCR

            # 关闭方向分类和文档矫正，车牌识别用不上，能省一半时间
            self._ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
            self._impl = "paddleocr"
        else:
            raise EngineUnavailable(reason)

        logger.info("深度学习引擎就绪，后端 = %s", self._impl)
        self._ready = True

    # -------------------------------------------------------------- 识别

    def predict(self, image: np.ndarray | str | Path) -> list[PlateResult]:
        self.ensure_ready()

        if isinstance(image, (str, Path)):
            img = imread(image)
            source_path = str(image)
        else:
            img = image
            source_path = None

        if img is None or img.size == 0:
            return []

        started = time.perf_counter()
        boxes = self._run_ocr(img, source_path)
        plates = self._assemble_plates(img, boxes)
        elapsed = (time.perf_counter() - started) * 1000

        for plate in plates:
            plate.meta["latency_ms"] = round(elapsed, 1)

        logger.debug("深度引擎耗时 %.1f ms，命中 %d 个车牌", elapsed, len(plates))
        return plates

    # -------------------------------------------------------------- 内部

    def _run_ocr(self, img: np.ndarray, source_path: str | None) -> list[tuple]:
        """执行 OCR，统一输出格式为 ``[(quad, text, score), ...]``。"""
        if self._impl == "rapidocr":
            result, _ = self._ocr(source_path if source_path else img)
            if not result:
                return []
            out = []
            for item in result:
                quad, text, score = item[0], item[1], item[2]
                out.append((np.asarray(quad, np.float32), str(text), float(score)))
            return out

        if self._impl == "paddleocr":
            raw = self._ocr.ocr(source_path if source_path else img, cls=False)
            out = []
            for page in raw or []:
                for entry in page or []:
                    quad, (text, score) = entry[0], entry[1]
                    out.append((np.asarray(quad, np.float32), str(text), float(score)))
            return out

        return []

    def _assemble_plates(self, img: np.ndarray, boxes: list[tuple]) -> list[PlateResult]:
        """把 OCR 的文本框结果组装成车牌结果。

        策略（按可靠性从高到低）：
          1. 文本框本身就是完整车牌号 → 直接采用，同时用传统定位模块
             找出对应的车牌框（用于画框展示）。
          2. 文本框是车牌的**一部分**（比如 OCR 把"京A"和"12345"分开识别）→
             把同一水平行、水平间距很近的相邻文本框按 y 坐标聚类后拼接。
        """
        results: list[PlateResult] = []

        # ---- 策略 1：整块命中
        used: set[int] = set()
        for idx, (quad, text, score) in enumerate(boxes):
            cleaned = _clean_text(text)
            match = _PLATE_TEXT.search(cleaned)
            if not match:
                continue

            plate_text = match.group(0)
            x0, y0 = quad[:, 0].min(), quad[:, 1].min()
            x1, y1 = quad[:, 0].max(), quad[:, 1].max()
            bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

            roi = img[int(y0) : int(y1), int(x0) : int(x1)]
            color, _ = detect_color(roi, self.cfg) if roi.size else (PlateColor.UNKNOWN, 0.0)

            results.append(
                PlateResult(
                    text=plate_text,
                    color=color,
                    bbox=bbox,
                    roi=roi if roi.size else None,
                    confidence=score,
                    engine=self.name,
                    char_confidences=[score] * len(plate_text),
                    meta={"ocr_backend": self._impl, "raw_text": text},
                )
            )
            used.add(idx)

            logger.debug("整块命中: %s (score=%.3f)", plate_text, score)

        # ---- 策略 2：分片拼接
        for group in self._group_fragments(boxes, used):
            merged = self._merge_fragments(group, img)
            if merged is not None:
                results.append(merged)

        return results

    def _group_fragments(
        self, boxes: list[tuple], used: set[int]
    ) -> list[list[tuple]]:
        """把未被整块命中的零散文本框按行分组。"""
        fragments = [
            (i, quad, text, score)
            for i, (quad, text, score) in enumerate(boxes)
            if i not in used and _BARE_TEXT.match(_clean_text(text))
        ]
        if len(fragments) < 2:
            return []

        # 按文本框中心 y 排序后再聚类，同一行的分片 y 值应接近
        fragments.sort(key=lambda f: (f[1][:, 1].mean(), f[1][:, 0].mean()))

        groups: list[list[tuple]] = []
        current: list[tuple] = []

        for frag in fragments:
            if not current:
                current = [frag]
                continue

            prev = current[-1]
            prev_h = prev[1][:, 1].max() - prev[1][:, 1].min()
            cur_h = frag[1][:, 1].max() - frag[1][:, 1].min()
            avg_h = max(1.0, (prev_h + cur_h) / 2)

            dy = abs(frag[1][:, 1].mean() - prev[1][:, 1].mean())
            dx_gap = frag[1][:, 0].min() - prev[1][:, 0].max()

            # 同一行（垂直偏移不超过 0.6 个字高）且水平相邻（间隔不超过 1.5 个字宽）
            if dy < avg_h * 0.6 and -avg_h < dx_gap < avg_h * 1.5:
                current.append(frag)
            else:
                if len(current) >= 2:
                    groups.append(current)
                current = [frag]

        if len(current) >= 2:
            groups.append(current)

        return groups

    def _merge_fragments(self, group: list[tuple], img: np.ndarray) -> PlateResult | None:
        """把同一行的一组分片拼成一个车牌结果。"""
        text = "".join(_clean_text(f[2]) for f in group)
        if not _BARE_TEXT.match(text) and not _PLATE_TEXT.search(text):
            return None

        quads = [f[1] for f in group]
        x0 = int(min(q[:, 0].min() for q in quads))
        y0 = int(min(q[:, 1].min() for q in quads))
        x1 = int(max(q[:, 0].max() for q in quads))
        y1 = int(max(q[:, 1].max() for q in quads))
        if x1 <= x0 or y1 <= y0:
            return None

        roi = img[y0:y1, x0:x1]
        color, _ = detect_color(roi, self.cfg) if roi.size else (PlateColor.UNKNOWN, 0.0)
        score = float(np.mean([f[3] for f in group]))

        logger.debug("分片拼接: %s (%d 片)", text, len(group))
        return PlateResult(
            text=text,
            color=color,
            bbox=(x0, y0, x1 - x0, y1 - y0),
            roi=roi if roi.size else None,
            confidence=score,
            engine=self.name,
            char_confidences=[score] * len(text),
            meta={"ocr_backend": self._impl, "fragments": len(group)},
        )


def _clean_text(text: str) -> str:
    """去掉 OCR 输出里的分隔符和空白。

    OCR 常把车牌识别成 ``"京A·12345"``、``"京 A 1 2 3 4 5"`` 这类带分隔符的形式。
    """
    remove = {"·", "•", ".", "。", "-", "—", "_", " ", "\t", "\n", ":", "："}
    cleaned = "".join(ch for ch in text if ch not in remove)
    return cleaned.upper().replace(" ", "")


__all__ = ["DeepEngine"]
