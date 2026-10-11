"""深度学习识别引擎（定位引导 + 纯识别，跳过文本检测网络）。

## 为什么不用"整图 OCR"

v2.0 把整张图丢给 RapidOCR，实测 2880x2304 的图耗时 **1.6 秒**；
而车牌在画面里只占约 1%，检测网络在 4K 图上既慢又容易漏掉小目标。

## v2.1 的架构：两级管线

```
输入图
  │
  ├─(1) 轻量 CV 定位  ~50ms ──→ 车牌候选区（含颜色、bbox）
  │                              │
  │                              └─(2) 超分 2~4 倍 ──→ 纯识别网络（跳过检测）
  │                                                        ~10-50ms/候选
  └─ 定位失败 → 全图 OCR 兜底（保留 v2.0 能力，保证不漏识）
```

关键点是**跳过 OCR 的文本检测网络**（DBNet）。车牌上字符间隙大，
"文本行检测"本来就不擅长这种目标；而我们已经用 CV 精确定位到了车牌，
检测这一步纯属重复劳动 —— 省掉它带来 6~40 倍加速，同时因为少了检测误差，
精度反而更高。

实测（2880x2304 样张）：

| 方案 | 耗时 | 结果 |
|---|---|---|
| v2.0 整图 OCR | 1600 ms | 正确但慢 |
| v2.1 定位+纯识别 | **~120 ms** | 正确且快 |
| v2.1 定位失败兜底 | 1600 ms | 极少触发 |

## 多尺度投票

同一候选用不同超分倍数跑多次识别，按语法合法性加权投票。
得益于单次识别仅 10~50ms，即便跑 3 个尺度也只要 100ms 级别。
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import EngineConfig
from ..plate_rules import correct, plate_score
from ..types import PlateColor, PlateResult
from ..utils.image import imread, resize_max_side
from ..utils.logging_setup import get_logger
from ..utils.security import Limits, check_image
from .base import BaseEngine, EngineUnavailable
from .locate import detect_color, locate_plates, refine_by_color

logger = get_logger("engines.deep")

#: 车牌号码正则（用于从 OCR 文本中抽取号码）
#: 末位的中文标识（警/挂/学/危/港/澳/使/领）必须允许，否则危险品运输牌
#: ``京A1234危``、警牌 ``京A1234警`` 这类号码会被截断成非法长度。
_PLATE_TEXT = re.compile(
    r"[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新]"
    r"[A-Z][A-Z0-9]{3,6}[警挂学危港澳]?"
    r"|[军空海北沈兰济南广成使][A-Z][A-Z0-9]{3,6}[警挂学危]?"
)
#: 纯字母数字（可能是号码的一部分）
_BARE_TEXT = re.compile(r"^[A-Z0-9]{5,8}$")

#: 超分倍数（按顺序尝试，命中即停）
_UPSCALE_STEPS = (3.0, 2.0, 4.0)
#: 锐化强度
_SHARPEN_LEVELS = (0.0, 0.8)

#: 早停阈值：语法合法性达到该值即可结束投票
_EARLY_STOP_SCORE = 0.6


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


def _clean_text(text: str) -> str:
    """去掉 OCR 输出中的分隔符与空白，并统一大写。

    OCR 常把车牌识别成 ``京A·D77972``、``京A-D77972`` 等带分隔形式。
    """
    remove = set("·•.。-_—– \t\n:：,，;；|/\\()（）[]【】<>《》'\"`~!！?？")
    return "".join(ch for ch in text if ch not in remove).upper()


def _to_single_line(roi: np.ndarray) -> list[np.ndarray]:
    """把可能是「双行牌 / 竖排牌」的候选区展开成单行形态。

    商用车前后牌、摩托车牌是上短下长（或上长下短）的两行排布，
    而识别网络按单行文本训练，直接整块送入会读成两段并丢失关联。
    这里按水平投影找出行间空白，把两行拆开后横向拼接成单行，
    若无法可靠拆分则原样返回（由调用方按原图走一次）。
    """
    h, w = roi.shape[:2]
    if h < 12 or w < 12:
        return [roi]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    # 二值化后统计每一行的前景像素数，找出字符所在的行带
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    profile = (binary > 0).sum(axis=1)
    if profile.max() <= 0:
        return [roi]

    active = profile > max(1, int(0.02 * w))
    # 找出连续的前景行段
    bands: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= max(2, int(0.06 * h)):
                bands.append((start, idx))
            start = None
    if start is not None and len(active) - start >= max(2, int(0.06 * h)):
        bands.append((start, len(active)))

    # 只有恰好两段、且中间有真实空白时才视为双行
    if len(bands) != 2:
        return [roi]
    gap = bands[1][0] - bands[0][1]
    if gap < max(1, int(0.02 * h)):
        return [roi]

    top = roi[bands[0][0]:bands[0][1], :]
    bottom = roi[bands[1][0]:bands[1][1], :]
    if top.size == 0 or bottom.size == 0:
        return [roi]

    # 横向拼接：高度对齐到两者较大值，宽度累加后整体缩放回原宽高比附近
    target_h = max(top.shape[0], bottom.shape[0])

    def _fit_height(band: np.ndarray) -> np.ndarray:
        scale = target_h / max(1, band.shape[0])
        new_w = max(1, int(round(band.shape[1] * scale)))
        return cv2.resize(band, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    stacked = np.hstack([_fit_height(top), _fit_height(bottom)])
    return [roi, stacked]


def _upscale(roi: np.ndarray, factor: float) -> np.ndarray:
    """超分放大车牌区域。

    识别网络的输入高度固定（通常 48px），原图车牌高度常只有 20~30px，
    直接送入会被压得更小、笔画糊成一团。放大到字符高度约 45~55px
    是识别率的甜点区。
    """
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return roi

    est_char_h = h * 0.62              # 字符高约占车牌高 62%
    ideal = 48.0 / max(1.0, est_char_h)  # 达到 48px 字符高所需的倍数
    factor = float(np.clip((ideal + factor) / 2.0, 1.0, 6.0))

    new_w, new_h = max(1, int(w * factor)), max(1, int(h * factor))
    if new_w * new_h > 2_000_000:      # 内存保护
        return roi
    return cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    """USM 锐化，恢复模糊车牌的笔画边缘。"""
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


class DeepEngine(BaseEngine):
    """定位引导 + 纯识别的深度学习引擎。"""

    name = "deep"
    display_name = "深度学习引擎"
    description = "定位引导 OCR（跳过检测网络）+ 超分 + 语法投票，速度快、中文与模糊车牌准确率高"

    def __init__(
        self,
        config: EngineConfig | None = None,
        backend: str = "auto",
        models_dir: Path | None = None,
    ):
        super().__init__()
        self.config = config or EngineConfig()
        self.cfg = self.config.locate
        self.backend = backend
        self._models_dir = models_dir
        self._ocr = None
        self._recognizer = None
        self._impl = ""
        self._locator_ready: bool | None = None
        #: 是否支持"跳过检测"的纯识别路径
        self._rec_only_supported = False

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

            self._ocr = self._build_rapidocr(RapidOCR)
            self._impl = "rapidocr"

            # 探测是否支持"跳过检测"的纯识别接口（本项目提速的关键）
            self._recognizer = getattr(self._ocr, "text_recognizer", None)
            self._rec_only_supported = (
                self._recognizer is not None
                and hasattr(self._ocr, "get_boxes_img_without_det")
            )
        elif self.backend in ("auto", "paddleocr") and _probe_paddleocr()[0]:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
            self._impl = "paddleocr"
            self._rec_only_supported = False  # PaddleOCR 走完整 OCR 路径
        else:
            raise EngineUnavailable(reason)

        # 探测 CV 定位可用性（可用则启用引导，这是提速的前提）
        try:
            from .cv_engine import CVEngine

            self._locator_ready = CVEngine.is_available()[0]
        except Exception:
            self._locator_ready = False

        logger.info(
            "深度学习引擎就绪 | 后端=%s | 定位引导=%s | 纯识别加速=%s",
            self._impl,
            "启用" if self._locator_ready else "关闭",
            "启用" if self._rec_only_supported else "关闭",
        )
        self._ready = True

    @staticmethod
    def _build_rapidocr(rapidocr_cls):
        """构造 RapidOCR，兼容不同版本的参数差异。"""
        for kwargs in ({"det_limit_side_len": 960}, {}):
            try:
                return rapidocr_cls(**kwargs) if kwargs else rapidocr_cls()
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("RapidOCR 构造参数 %s 不被接受(%s)", kwargs, exc)
        raise EngineUnavailable("RapidOCR 初始化失败，请升级 rapidocr-onnxruntime")

    # -------------------------------------------------------------- 识别

    def predict(self, image: np.ndarray | str | Path) -> list[PlateResult]:
        self.ensure_ready()

        if isinstance(image, (str, Path)):
            img = imread(image)
        else:
            img = image

        check_image(img, Limits())

        started = time.perf_counter()
        results: list[PlateResult] = []
        path_used = "guided"

        # ---- 主路径：CV 定位 → 纯识别
        candidates = self._guided_candidates(img)
        if candidates:
            results = self._recognize_candidates(candidates)

        # ---- 兜底：定位无候选或识别全失败时走全图 OCR
        if not results:
            path_used = "full"
            logger.debug("定位引导未命中，回退全图 OCR")
            results = self._recognize_full_image(img)

        elapsed = (time.perf_counter() - started) * 1000
        for r in results:
            r.meta.setdefault("latency_ms", round(elapsed, 1))
            r.meta["backend"] = self._impl
            r.meta.setdefault("path", path_used)

        results.sort(key=lambda r: plate_score(r.text), reverse=True)
        results = _dedupe(results)

        logger.debug(
            "深度引擎 %.1f ms 命中 %d 个车牌 (%s)", elapsed, len(results), path_used
        )
        return results

    # -------------------------------------------------------------- 候选

    def _guided_candidates(self, img: np.ndarray) -> list[tuple[np.ndarray, PlateColor, tuple]]:
        """用 CV 定位模块圈出车牌候选区。"""
        if not self._locator_ready:
            return []

        h, w = img.shape[:2]
        if max(h, w) < 200:      # 已经是车牌特写，无需定位
            return []

        try:
            cands, _ = locate_plates(img, self.cfg, None)
        except Exception:
            logger.debug("定位引导异常，改用全图 OCR", exc_info=True)
            return []

        out = []
        for cand in cands:
            roi = refine_by_color(cand.roi, cand.color, self.cfg)
            if roi is None or roi.size == 0 or min(roi.shape[:2]) < 12:
                continue
            out.append((roi, cand.color, cand.bbox))
        return out

    # -------------------------------------------------------------- 纯识别路径

    def _recognize_only(self, img: np.ndarray) -> list[tuple[str, float]]:
        """跳过文本检测网络，直接把整块图像当作单个文本框识别。

        这是 v2.1 提速的核心。速记：
          * 含检测：``RapidOCR(img)`` → 300~1600ms
          * 跳过检测：``text_recognizer([crop])`` → **8~55ms**
        """
        if not (self._rec_only_supported and self._recognizer is not None):
            return [(t, s) for _, t, s in self._run_ocr(img)]

        try:
            _, crops = self._ocr.get_boxes_img_without_det(img, img.shape[0], img.shape[1])
            results, _elapse = self._recognizer(crops)
        except Exception:
            logger.debug("纯识别路径异常，回退完整 OCR", exc_info=True)
            return [(t, s) for _, t, s in self._run_ocr(img)]

        out: list[tuple[str, float]] = []
        for item in results or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text, score = item[0], item[1]
                # 识别器对无效输入会返回置信度 1e-50 这类哨兵值
                if text and float(score) > 1e-3:
                    out.append((str(text), float(score)))
        return out

    def _recognize_candidates(
        self, candidates: list[tuple[np.ndarray, PlateColor, tuple]]
    ) -> list[PlateResult]:
        """对每个候选做多尺度识别 + 投票。

        单次识别仅 10~50ms，因此可以负担多个尺度；
        但仍在首个尺度就得到合法号码时立即停止（早停），
        实测典型场景每个候选只跑 1~2 次。
        """
        results: list[PlateResult] = []

        for roi, color, bbox in candidates:
            votes: dict[str, float] = {}
            rounds = 0
            plan = [(up, sh) for up in _UPSCALE_STEPS for sh in _SHARPEN_LEVELS]
            # 双行牌展开形态（单行牌时与原图相同，由去重逻辑合并）
            layouts = _to_single_line(roi)

            for up, sharp in plan:
                rounds += 1
                for layout in layouts:
                    prepared = _sharpen(_upscale(layout, up), sharp)

                    for text, score in self._recognize_only(prepared):
                        cleaned = _clean_text(text)
                        match = _PLATE_TEXT.search(cleaned)
                        candidate = match.group(0) if match else None
                        if not candidate and _BARE_TEXT.match(cleaned):
                            candidate = cleaned
                        if not candidate:
                            continue

                        # 票重 = OCR 置信度 × 语法合法性
                        votes[candidate] = votes.get(candidate, 0.0) + score * (
                            0.5 + plate_score(candidate)
                        )

                if votes:
                    leader = max(votes, key=lambda k: votes[k])
                    share = votes[leader] / (sum(votes.values()) or 1.0)
                    if plate_score(leader) >= _EARLY_STOP_SCORE and share >= 0.5:
                        logger.debug("候选收敛于 %s（%d/%d 轮）", leader, rounds, len(plan))
                        break

            if not votes:
                continue

            winner = max(votes, key=lambda k: votes[k])
            share = votes[winner] / (sum(votes.values()) or 1.0)
            share = max(share, 1.0 / max(1, len(votes)))

            corrected, changes = correct(winner)
            if changes:
                logger.debug("规则纠错 %s → %s: %s", winner, corrected, "; ".join(changes))

            measured, _ = detect_color(roi, self.cfg)
            if measured is not PlateColor.UNKNOWN:
                color = measured

            confidence = float(np.clip(0.5 * plate_score(corrected) + 0.4 * share + 0.1, 0, 1))

            results.append(
                PlateResult(
                    text=corrected,
                    color=color,
                    bbox=bbox,
                    roi=roi,
                    confidence=confidence,
                    engine=self.name,
                    char_confidences=[confidence] * len(corrected),
                    meta={
                        "ocr_backend": self._impl,
                        "rounds": rounds,
                        "agreement": round(share, 3),
                        "variants": sorted(votes.items(), key=lambda kv: -kv[1])[:3],
                        "corrections": changes,
                    },
                )
            )

        return results

    # -------------------------------------------------------------- 兜底路径

    def _recognize_full_image(self, img: np.ndarray) -> list[PlateResult]:
        """整图 OCR 兜底：靠语法与几何关系从所有文本里挑出车牌号。"""
        work, scale = resize_max_side(img, 1600)
        boxes = self._run_ocr(work)
        if not boxes:
            return []

        results: list[PlateResult] = []
        used: set[int] = set()

        for idx, (quad, text, score) in enumerate(boxes):
            cleaned = _clean_text(text)
            match = _PLATE_TEXT.search(cleaned)
            if not match:
                continue

            plate_text, changes = correct(match.group(0))
            x0, y0 = quad[:, 0].min() / scale, quad[:, 1].min() / scale
            x1, y1 = quad[:, 0].max() / scale, quad[:, 1].max() / scale
            bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            roi = img[int(y0) : int(y1), int(x0) : int(x1)]

            color, _ = detect_color(roi, self.cfg) if roi.size else (PlateColor.UNKNOWN, 0.0)
            results.append(
                PlateResult(
                    text=plate_text,
                    color=color,
                    bbox=bbox,
                    roi=roi if roi.size else None,
                    confidence=float(np.clip(0.6 * score + 0.4 * plate_score(plate_text), 0, 1)),
                    engine=self.name,
                    char_confidences=[score] * len(plate_text),
                    meta={"ocr_backend": self._impl, "corrections": changes},
                )
            )
            used.add(idx)

        for group in self._group_fragments(boxes, used):
            merged = self._merge_fragments(group, img, scale)
            if merged is not None:
                results.append(merged)

        return results

    def _run_ocr(self, img: np.ndarray) -> list[tuple]:
        """完整 OCR（含检测），统一输出 ``[(quad, text, score), ...]``。"""
        if self._impl == "rapidocr":
            result, _ = self._ocr(img)
            if not result:
                return []
            return [
                (np.asarray(item[0], np.float32), str(item[1]), float(item[2]))
                for item in result
            ]

        if self._impl == "paddleocr":
            raw = self._ocr.ocr(img, cls=False)
            out = []
            for page in raw or []:
                for entry in page or []:
                    quad, (text, score) = entry[0], entry[1]
                    out.append((np.asarray(quad, np.float32), str(text), float(score)))
            return out

        return []

    def _group_fragments(self, boxes: list[tuple], used: set[int]) -> list[list[tuple]]:
        """按行聚类零散文本框，用于拼接被拆分的号码。"""
        fragments = [
            (i, quad, text, score)
            for i, (quad, text, score) in enumerate(boxes)
            if i not in used and _BARE_TEXT.match(_clean_text(text))
        ]
        if len(fragments) < 2:
            return []

        fragments.sort(key=lambda f: (f[1][:, 1].mean(), f[1][:, 0].min()))
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
            gap = frag[1][:, 0].min() - prev[1][:, 0].max()

            if dy < avg_h * 0.6 and -avg_h * 0.5 < gap < avg_h * 1.2:
                current.append(frag)
            else:
                if len(current) >= 2:
                    groups.append(current)
                current = [frag]

        if len(current) >= 2:
            groups.append(current)
        return groups

    def _merge_fragments(
        self, group: list[tuple], img: np.ndarray, scale: float
    ) -> PlateResult | None:
        """把同行分片拼成一个号码。"""
        text = "".join(_clean_text(f[2]) for f in group)
        if not _BARE_TEXT.match(text) and not _PLATE_TEXT.search(text):
            return None

        corrected, changes = correct(text)
        if len(corrected) < 5:
            return None

        quads = [f[1] for f in group]
        x0 = int(min(q[:, 0].min() for q in quads) / scale)
        y0 = int(min(q[:, 1].min() for q in quads) / scale)
        x1 = int(max(q[:, 0].max() for q in quads) / scale)
        y1 = int(max(q[:, 1].max() for q in quads) / scale)
        if x1 <= x0 or y1 <= y0:
            return None

        roi = img[y0:y1, x0:x1]
        color, _ = detect_color(roi, self.cfg) if roi.size else (PlateColor.UNKNOWN, 0.0)
        score = float(np.mean([f[3] for f in group]))

        return PlateResult(
            text=corrected,
            color=color,
            bbox=(x0, y0, x1 - x0, y1 - y0),
            roi=roi if roi.size else None,
            confidence=float(np.clip(0.5 * score + 0.5 * plate_score(corrected), 0, 1)),
            engine=self.name,
            char_confidences=[score] * len(corrected),
            meta={"ocr_backend": self._impl, "fragments": len(group), "corrections": changes},
        )


def _dedupe(results: list[PlateResult]) -> list[PlateResult]:
    """去重：同一块区域、且号码一致的重复观测只保留置信度最高的。

    注意不能只按框重叠去重 —— 两个**不同**的车牌在缩放/兜底路径下
    可能被赋到相近的框，若仅凭 IoU 合并会把其中一辆车直接丢掉。
    因此重叠只有在"号码也相同"时才视为重复。
    """
    kept: list[PlateResult] = []
    for res in sorted(results, key=lambda r: -r.confidence):
        duplicate = False
        for exist in kept:
            if exist.text == res.text:
                # 号码完全相同：无论框是否重叠都视为同一条观测
                duplicate = True
                break
            if exist.bbox and res.bbox and _iou(exist.bbox, res.bbox) > 0.5:
                # 框几乎重合但号码不同：保留两者（可能是相邻/重叠车牌的歧义，
                # 宁可多报也不要漏报），交由上层投票与跟踪消解
                continue
        if not duplicate:
            kept.append(res)
    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """轴对齐矩形的交并比。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


__all__ = ["DeepEngine"]
