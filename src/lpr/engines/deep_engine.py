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
from ..plate_rules import correct, is_valid, loose_match, plate_score
from ..types import PlateColor, PlateLayout, PlateResult
from ..utils.image import ensure_bgr, imread, resize_max_side
from ..utils.logging_setup import get_logger
from ..utils.security import Limits, check_image
from .base import BaseEngine, EngineUnavailable
from .locate import detect_color, locate_plates, refine_by_color

logger = get_logger("engines.deep")

#: 车牌号码正则（用于从 OCR 文本中抽取号码）—— **严格层**
#: 末位的中文标识（警/挂/学/危/港/澳/使/领）必须允许，否则危险品运输牌
#: ``京A1234危``、警牌 ``京A1234警`` 这类号码会被截断成非法长度。
#: 「使/领」此前不在末位可选集里，使领馆牌 ``京123使`` 会被截成
#: ``京123``（6 位残串，既不合法也丢了后缀），已补上。
#: 武警牌（``WJ京1234X`` / 旧式 ``WJ0112345``）与「使 + 数字」的使领馆牌
#: 此前完全没有分支，严格层抽不到、只能靠宽松召回兜底，已补齐。
#: 民航号牌以"民航"两字开头（如 民航A0001），既不是省份简称也不是军队
#: 字头，必须单独开一个分支，否则 OCR 抽不出这段文本。
#:
#: 与 ``plate_rules.LOOSE_STRUCTURE`` 的分工：本正则**首选**使用（更精确，
#: 能直接筛掉非车牌文本）；抽不到时再退到宽松召回（允许首位形近字）。
_PLATE_TEXT = re.compile(
    r"[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新]"
    r"[A-Z][A-Z0-9]{3,6}[警挂学危港澳使领]?"
    r"|[军空海北沈兰济南广成][A-Z][A-Z0-9]{3,6}[警挂学危]?"
    r"|使[0-9]{4,6}"
    r"|[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新][0-9]{3}[使领]"
    r"|WJ[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新][0-9]{4}[A-HJ-NP-Z0-9]?"
    r"|WJ[0-9]{5,7}"
    r"|民航[A-G][0-9]{3,4}"
)
#: 纯字母数字（可能是号码的一部分）
_BARE_TEXT = re.compile(r"^[A-Z0-9]{5,8}$")

#: 超分倍数（按顺序尝试，命中即停）
_UPSCALE_STEPS = (3.0, 2.0, 4.0)
#: 锐化强度
_SHARPEN_LEVELS = (0.0, 0.8)

#: 早停阈值：语法合法性达到该值即可结束投票
_EARLY_STOP_SCORE = 0.6

#: 判定"双层牌"时允许的最大宽高比。
#: 单行牌 440x140mm（≈3.14:1），双行牌 440x220mm（≈2.0:1）。
#: 取 2.35 作为上限：再宽就不可能是双行牌，那两段"行带"是边框而非两行字符。
_DOUBLE_MAX_ASPECT = 2.35

#: 计算行投影时左右各裁掉的比例（去掉车牌竖直边框线）。
#: 不裁的话，框线会在每一行都贡献前景像素，把上下两行字符桥接成一段。
_BORDER_TRIM = 0.08

#: 行带桥接容忍比例：宽度小于该比例的空白视为噪声填掉
_BAND_BRIDGE_RATIO = 0.012

#: 单行牌防线：某段行带高度占比超过该值 → 判定为单行字符，不拆
_SINGLE_BAND_RATIO = 0.50

#: 双行牌两行行带的高度相似度下限（min/max）。
#: 真实双行牌上下行字号一致；上薄下厚多半是单行牌上方的装饰条。
_BAND_HEIGHT_SIMILARITY = 0.55


def _extract_plate(cleaned: str) -> str | None:
    """从清洗后的 OCR 文本里抽出一段车牌号。

    四级策略（借鉴参考实现 YOLO26-plate 的"宽松召回 / 严格校验"分层）：

    1. **严格召回**：用 :data:`_PLATE_TEXT` 找结构合法的片段（首选）。
       它要求首位是真实省份简称，能直接筛掉大量非车牌文本。
    2. **宽松召回**：用 :func:`loose_match` —— 允许首位是**省份形近字**
       （凉/泸/律…），把"结构对但首位认错"的号码也捞出来。
       此前这类号码会在抽取阶段就被丢弃，纠错逻辑根本没机会执行。
    3. **先纠正再召回**：把整串先跑一遍 :func:`correct`（如把 ``京三`` 的
       "三"修成字母 E），再走一次宽松召回 —— 专救"中间位被读成汉字"的情况。
    4. **纯字母数字兜底**：整串都是 ``[A-Z0-9]`` 时（首位省份被完全读丢，
       如 ``A12345``），仅当**纠正后能变成合法车牌**才接受。

    第 4 步的附加条件很关键：此前只判 ``_BARE_TEXT`` 就接受，导致全图 OCR
    把车身贴纸、广告字（``JINCBFI``、``A5331``）也当成车牌报出来。
    加一道"纠正后必须合法"的闸门后，这类噪声会被自然滤掉。
    """
    if not cleaned:
        return None

    match = _PLATE_TEXT.search(cleaned)
    if match:
        return match.group(0)

    loose = loose_match(cleaned)
    if loose:
        return loose

    # 先纠正再召回：修掉形如"京三A5331"（三→E）这类汉字误读后再试
    corrected, _changes = correct(cleaned)
    if corrected != cleaned:
        match = _PLATE_TEXT.search(corrected)
        if match:
            return match.group(0)
        loose = loose_match(corrected)
        if loose:
            return loose

    # 整串不是车牌，但后半段可能是（前面粘了无关文字/汉字）
    if len(cleaned) > 2:
        loose = loose_match(cleaned, start=1)
        if loose:
            return loose

    # 纯字母数字兜底：必须"纠正后合法"才接受，否则视为噪声
    if _BARE_TEXT.match(cleaned) and is_valid(correct(cleaned)[0]):
        return cleaned

    return None


def _accumulate_char_probs(
    store: dict[str, list[float]], text: str, probs: list[float]
) -> None:
    """把一次识别的逐字符置信度并入累积表（逐位取最大值）。"""
    if not text or not probs:
        return
    existing = store.get(text)
    if existing is None or len(existing) != len(probs):
        store[text] = list(probs)
        return
    store[text] = [max(a, b) for a, b in zip(existing, probs)]


def _align_char_probs(
    probs: list[float] | None,
    observed: str,
    final: str,
    fallback: float,
) -> list[float]:
    """把观测串的逐位置信度对齐到最终号码上。

    纠错可能改变字符（``凉→京``）或长度，对不上时按位置沿用或退化为
    整体置信度 —— 只保证长度与 ``final`` 一致，不臆造精度。
    """
    if not final:
        return []
    if not probs:
        return [fallback] * len(final)
    if len(probs) == len(final):
        return list(probs)
    if len(probs) == len(observed) and len(observed) == len(final):
        return list(probs)
    # 长度变了：保留共同前缀的逐位值，其余用整体置信度填充
    n = min(len(probs), len(final))
    return list(probs[:n]) + [fallback] * (len(final) - n)


def _weak_positions(probs: list[float], threshold: float = 0.7) -> list[int]:
    """返回置信度低于阈值的字符下标（0-based），供人工复核定位。"""
    return [i for i, p in enumerate(probs) if p < threshold]


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
    """把「双层（双行）牌」展开成单行形态。

    双行牌是上短下长的两行排布（如大型汽车后号牌：上行 4 字、下行 6 字），
    而识别网络按单行文本训练，直接整块送入会把两段读成互不相关的文本。
    这里把两行拆开后横向拼接成单行，让识别网络当作一行 10 字来处理。

    拼接口径（与参考实现 YOLO26-plate 的 ``get_split_merge`` 一致）：
    两行**先统一到同一高度再并排**，这样字符大小一致、符合识别网络的
    训练分布。若直接按原尺寸拼接，行高较矮的那行会被严重拉伸。

    返回值的第一项恒为原图，第二项（若存在）为拼接后的单行图。
    判断"是否为双层"请用 :func:`split_double_line`，不要靠本函数的长度。
    """
    # 复用 split_double_line 的切分结果，避免两处逻辑各写一份导致不一致
    pair = split_double_line(roi)
    if pair is None:
        return [roi]

    top, bottom = pair
    # 统一行高：取两行中较高的那个作为目标高度
    target_h = max(top.shape[0], bottom.shape[0])

    def _fit_height(band: np.ndarray) -> np.ndarray:
        scale = target_h / max(1, band.shape[0])
        new_w = max(1, int(round(band.shape[1] * scale)))
        return cv2.resize(band, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    stacked = np.hstack([_fit_height(top), _fit_height(bottom)])
    # 拼接后高宽比若已达单行牌特征（≥2.5:1），说明两行都是有效字符行
    return [roi, stacked]


def _band_profile(roi: np.ndarray) -> np.ndarray:
    """算出行方向的前景像素投影，用于定位字符行带。

    关键细节：**先裁掉左右各 8% 的边框**。车牌四周有 1~2px 的白色/黑色
    框线，它在每一行都会贡献前景像素，会把上下两行字符"桥接"成一段连续
    行带，导致永远切不出两段。

    （参考实现 we0091234/YOLO26-plate 直接按固定比例 5/12、1/3 切分来
    绕开这个问题；我们改成"裁边框 + 投影"的通用做法，不依赖固定比例，
    对各种宽高比的输入都成立。）
    """
    h, w = roi.shape[:2]
    margin = int(round(w * _BORDER_TRIM))
    if w - 2 * margin >= 0.4 * w:
        core = roi[:, margin : w - margin]
    else:
        core = roi

    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY) if core.ndim == 3 else core
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (binary > 0).sum(axis=1)


def _find_bands(profile: np.ndarray, h: int, width: int) -> list[tuple[int, int]]:
    """从投影里找出字符行带，并容忍被边框桥接的小缺口。

    单行牌字符高度可达车牌高度的 60% 以上，若某段行带占比超过
    ``_SINGLE_BAND_RATIO``，说明它已经是一整行字符（或两行被桥接在一起），
    此时不再往下切。这是"带边框单行牌被误拆"的最后一道防线。
    """
    if profile.size == 0 or profile.max() <= 0:
        return []

    active = profile > max(1, int(0.02 * width))
    # 桥接：把宽度小于 min_gap 的空白当噪声填掉（框线的抗锯齿/断开）
    min_gap = max(1, int(round(h * _BAND_BRIDGE_RATIO)))
    filled = active.copy()
    idx = 0
    n = len(active)
    while idx < n:
        if not active[idx]:
            run = idx
            while run < n and not active[run]:
                run += 1
            if idx > 0 and run < n and (run - idx) <= min_gap:
                filled[idx:run] = True
            idx = run
        else:
            idx += 1

    bands: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(filled):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= max(2, int(0.06 * h)):
                bands.append((start, i))
            start = None
    if start is not None and len(filled) - start >= max(2, int(0.06 * h)):
        bands.append((start, len(filled)))

    # 单行牌防线：占比过大的行带说明是单行字符，不拆
    result = []
    for band in bands:
        if (band[1] - band[0]) > h * _SINGLE_BAND_RATIO:
            return []
        result.append(band)
    return result


def split_double_line(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """判断候选区是否是「双层（双行）」车牌，是则返回上下两条行带。

    判据有三条，**全部**满足才算双层：

    1. 裁掉左右边框后，水平投影上恰好存在 2 段字符行带，且中间有空白间隔；
    2. 两段行带的**高度相近**（真实双行牌上下两行字号一致；若上薄下厚，
       多半是单行牌上方的装饰条被切出来了）；
    3. 整块区域的宽高比 ≤ ``_DOUBLE_MAX_ASPECT`` —— 双行牌 440x220mm 约 2:1，
       单行牌 440x140mm 约 3.1:1。宽高比过大的不可能是双行牌。

    这三条缺一不可：早期只按行带数量判断时，带上下边框的单行牌会被误判成
    双层并拆成乱码；而只看投影又会因为边框桥接而永远切不出两段。

    Returns:
        ``(上行, 下行)`` 图块，或 ``None``（不是双层牌）。
    """
    roi = ensure_bgr(roi, min_side=12)
    if roi is None:
        return None
    h, w = roi.shape[:2]

    # 宽高比过大 → 不可能是双行牌（双行牌高度约为宽度的一半）
    if w / max(1, h) > _DOUBLE_MAX_ASPECT:
        return None

    profile = _band_profile(roi)
    bands = _find_bands(profile, h, w)
    if len(bands) != 2:
        return None

    gap = bands[1][0] - bands[0][1]
    if gap < max(1, int(0.02 * h)):
        return None

    # 行带高度相近性检查：双行牌两行字号一致
    h0 = bands[0][1] - bands[0][0]
    h1 = bands[1][1] - bands[1][0]
    ratio = min(h0, h1) / max(1, max(h0, h1))
    if ratio < _BAND_HEIGHT_SIMILARITY:
        return None

    top = roi[bands[0][0]:bands[0][1], :]
    bottom = roi[bands[1][0]:bands[1][1], :]
    if top.size == 0 or bottom.size == 0:
        return None
    return top, bottom


def measure_layout(roi: np.ndarray, cfg: EngineConfig | None = None) -> str:
    """从车牌图像判定版式（单行 / 双层）。

    优先按行分割证据判定；若行分割不可靠，退化为**宽高比**先验：
    单行牌 440x140mm（≈3.14:1），双行牌 440x220mm（≈2:1）。
    500x400 放大倍数下的经验阈值取 2.6。

    Returns:
        ``"single"`` / ``"double"`` / ``"unknown"``（供 ``PlateLayout`` 解析）
    """
    if roi is None or roi.size == 0:
        return "unknown"

    roi = ensure_bgr(roi, min_side=8)
    if roi is None:
        return "unknown"

    h, w = roi.shape[:2]
    if h < 8 or w < 8:
        return "unknown"

    # ---- 证据 1（主导）：行分割
    # 它能直接切出两条高度相近、上下分离的行带 —— 这是双行牌最硬的证据。
    if split_double_line(roi) is not None:
        return "double"

    # 行分割"明确否定"：出现了内容过高（超过 50%）的单条行带。
    # 那事实上就是一行字符，比宽高比先验可靠得多，直接判单行。
    if _has_single_band(roi):
        return "single"

    # ---- 证据 2（辅助）：宽高比先验
    # 单行牌 440x140mm（≈3.14:1），双行牌 440x220mm（≈2.0:1）。
    # 仅在行分割拿不到结论时才用；且**只用于判单行**，不用于判双行 ——
    # 因为"矮胖"也可能是误检到的车身/阴影区域，判成 double 会让它白跑
    # 一遍双行拼接路径（实测 single_blue.jpg 的 602x277 车身区域曾被误判）。
    aspect = w / max(1, h)
    if aspect >= 2.6:
        return "single"
    return "unknown"


def _has_single_band(roi: np.ndarray) -> bool:
    """行投影里是否存在"内容过高、显然是一整行字符"的行带。

    ``_find_bands`` 遇到这种行带会返回空列表（单行牌防线），
    但"空列表"也可能是因为完全找不到行带（噪声图）。
    这里显式区分两者：只有**先找到行带、又因过高被拒**才算明确否定。
    """
    roi = ensure_bgr(roi, min_side=12)
    if roi is None:
        return False
    h, w = roi.shape[:2]
    profile = _band_profile(roi)
    if profile.size == 0 or profile.max() <= 0:
        return False

    active = profile > max(1, int(0.02 * w))
    bands: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= max(2, int(0.06 * h)):
                bands.append((start, i))
            start = None
    if start is not None and len(active) - start >= max(2, int(0.06 * h)):
        bands.append((start, len(active)))

    # 存在占比 >50% 的行带 → 那就是一整行字符
    return any((b[1] - b[0]) > h * _SINGLE_BAND_RATIO for b in bands)


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
        return [(t, s) for t, s, _ in self._recognize_only_detailed(img)]

    def _recognize_only_detailed(
        self, img: np.ndarray
    ) -> list[tuple[str, float, list[float]]]:
        """同 :meth:`_recognize_only`，但额外返回**逐字符置信度**。

        逐字符置信度是定位"哪个字认错了"的关键信息：整串 0.95 看着很好，
        但可能是 ``京A1**45`` 里两位只有 0.3。参考实现 YOLO26-plate 的
        ``get_plate_result`` 同样保留每字符概率（``prob[new_index]``），
        我们也把这份信息带出来，用于：
          * 输出审计（哪个位置不可信）
          * 低置信字符触发的二次识别
        """
        if not (self._rec_only_supported and self._recognizer is not None):
            return [(t, s, [s] * len(t)) for _, t, s in self._run_ocr(img)]

        try:
            _, crops = self._ocr.get_boxes_img_without_det(img, img.shape[0], img.shape[1])
            results, _elapse = self._recognizer(crops)
        except Exception:
            logger.debug("纯识别路径异常，回退完整 OCR", exc_info=True)
            return [(t, s, [s] * len(t)) for _, t, s in self._run_ocr(img)]

        out: list[tuple[str, float, list[float]]] = []
        for item in results or []:
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue
            text, score = item[0], item[1]
            # 识别器对无效输入会返回置信度 1e-50 这类哨兵值
            if not text or float(score) <= 1e-3:
                continue
            char_probs = self._extract_char_probs(item, str(text), float(score))
            out.append((str(text), float(score), char_probs))
        return out

    @staticmethod
    def _extract_char_probs(item: tuple, text: str, fallback: float) -> list[float]:
        """从识别器输出里取出逐字符置信度，取不到就退化为整串分数。

        RapidOCR 不同版本的第 3 个元素形态不一（可能是逐字概率列表，
        也可能没有）。这里只做"有则用、无则退化"的稳健处理，不去猜结构。
        """
        if len(item) >= 3:
            third = item[2]
            if isinstance(third, (list, tuple, np.ndarray)) and len(third) == len(text):
                try:
                    probs = [float(v) for v in third]
                    if all(0.0 <= p <= 1.0 for p in probs):
                        return probs
                except (TypeError, ValueError):
                    pass
        return [fallback] * len(text)

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
            #: 每个候选串累积的逐字符置信度（多轮识别取逐位最大值）
            char_probs_by_text: dict[str, list[float]] = {}
            rounds = 0
            plan = [(up, sh) for up in _UPSCALE_STEPS for sh in _SHARPEN_LEVELS]
            # 双行牌展开形态。
            # 借鉴参考实现 YOLO26-plate 的思路：**先在检测/定位阶段判定
            # 单层还是双层**，而不是对每个候选无脑尝试拆分。
            # 这里用 measure_layout 做等价判定——只有确实是双层的候选
            # 才生成拼接形态，单行牌只走原图，避免两倍无效识别开销。
            layout = PlateLayout(measure_layout(roi))
            variants = _to_single_line(roi) if layout is PlateLayout.DOUBLE else [roi]

            for up, sharp in plan:
                rounds += 1
                for variant in variants:
                    prepared = _sharpen(_upscale(variant, up), sharp)

                    for text, score, char_probs in self._recognize_only_detailed(prepared):
                        cleaned = _clean_text(text)
                        candidate = _extract_plate(cleaned)
                        if not candidate:
                            continue

                        # 票重 = OCR 置信度 × 语法合法性
                        # 合法候选额外加权：让"结构合法"的读数压过"结构不合法"的
                        votes[candidate] = votes.get(candidate, 0.0) + score * (
                            0.5 + plate_score(candidate)
                        )
                        # 累积逐字符置信度（用于输出审计与低置信二次识别）
                        _accumulate_char_probs(char_probs_by_text, candidate, char_probs)

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

            # layout 已在候选阶段算好（见上方 measure_layout 调用）

            confidence = float(np.clip(0.5 * plate_score(corrected) + 0.4 * share + 0.1, 0, 1))

            # 逐字符置信度：把投票时累积的值对齐到**纠正后**的号码上。
            # 纠错可能改动了字符（如 凉→京），此时置信度按位置沿用原值，
            # 便于定位"哪个位置不可信"。
            char_confs = _align_char_probs(
                char_probs_by_text.get(winner), winner, corrected, confidence
            )

            results.append(
                PlateResult(
                    text=corrected,
                    color=color,
                    layout=layout,
                    bbox=bbox,
                    roi=roi,
                    confidence=confidence,
                    engine=self.name,
                    char_confidences=char_confs,
                    meta={
                        "ocr_backend": self._impl,
                        "rounds": rounds,
                        "agreement": round(share, 3),
                        "variants": sorted(votes.items(), key=lambda kv: -kv[1])[:3],
                        "corrections": changes,
                        "aspect": round(roi.shape[1] / max(1, roi.shape[0]), 2),
                        "split_double": layout is PlateLayout.DOUBLE,
                        # 最不可信的字符位置（0-based），便于人工复核与调参
                        "weak_chars": _weak_positions(char_confs),
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
            # 与候选路径同一套「严格优先 + 宽松兜底」抽取逻辑
            raw = _extract_plate(cleaned)
            if not raw:
                continue

            plate_text, changes = correct(raw)
            x0, y0 = quad[:, 0].min() / scale, quad[:, 1].min() / scale
            x1, y1 = quad[:, 0].max() / scale, quad[:, 1].max() / scale
            bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            roi = img[int(y0) : int(y1), int(x0) : int(x1)]

            color, _ = detect_color(roi, self.cfg) if roi.size else (PlateColor.UNKNOWN, 0.0)
            results.append(
                PlateResult(
                    text=plate_text,
                    color=color,
                    layout=PlateLayout(measure_layout(roi)) if roi.size else PlateLayout.UNKNOWN,
                    bbox=bbox,
                    roi=roi if roi.size else None,
                    confidence=float(np.clip(0.6 * score + 0.4 * plate_score(plate_text), 0, 1)),
                    engine=self.name,
                    char_confidences=[score] * len(plate_text),
                    meta={
                        "ocr_backend": self._impl,
                        "corrections": changes,
                        "path": "full",
                    },
                )
            )
            used.add(idx)

        for group in self._group_fragments(boxes, used):
            merged = self._merge_fragments(group, img, scale)
            if merged is not None:
                results.append(merged)
            else:
                used.discard(group[0][0])  # 归还，供垂直合并复用

        # 双层牌垂直合并：上牌行 + 下牌行 在水平方向重叠、垂直方向相邻
        for group in self._group_double_fragments(boxes, used):
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

    def _group_double_fragments(
        self, boxes: list[tuple], used: set[int]
    ) -> list[list[tuple]]:
        """识别**上下排布**的双层牌碎片：上行（省份+发牌机关）+ 下行（序号）。

        与 :meth:`_group_fragments` 的同行水平拼接不同，双层牌的两个
        碎片在**垂直方向相邻、水平方向重叠**，如::

            京E      ← 上行（可能被误读成"京三"）
           A5331     ← 下行

        判定条件（从严，防止把车厢上不相关的两行文字拼起来）：
          * 两碎片水平方向的重叠量 ≥ 较窄一方宽度的 40%
          * 垂直间隙在 0.2~1.2 倍平均行高之间（紧贴但不分离）
          * 两行高度相近（比值 0.5~2.0），符合真实双层牌版式
          * 上下拼接后能被 :func:`_extract_plate` 抽出车牌，且纠正后合法
        """
        fragments = [
            (i, quad, text, score)
            for i, (quad, text, score) in enumerate(boxes)
            if i not in used and len(_clean_text(text)) <= 6
        ]
        if len(fragments) < 2:
            return []

        groups: list[list[tuple]] = []
        taken: set[int] = set()

        for a in fragments:
            if a[0] in taken:
                continue
            for b in fragments:
                if b[0] in taken or b[0] == a[0]:
                    continue
                top, bot = (a, b) if a[1][:, 1].mean() <= b[1][:, 1].mean() else (b, a)
                ta, tb = _clean_text(top[2]), _clean_text(bot[2])
                if not ta or not tb:
                    continue

                th = top[1][:, 1].max() - top[1][:, 1].min()
                bh = bot[1][:, 1].max() - bot[1][:, 1].min()
                avg_h = max(1.0, (th + bh) / 2)
                if not (0.5 <= th / max(1.0, bh) <= 2.0):
                    continue

                # 水平重叠量
                lo = max(top[1][:, 0].min(), bot[1][:, 0].min())
                hi = min(top[1][:, 0].max(), bot[1][:, 0].max())
                overlap = hi - lo
                narrow = min(
                    top[1][:, 0].max() - top[1][:, 0].min(),
                    bot[1][:, 0].max() - bot[1][:, 0].min(),
                )
                if overlap < narrow * 0.4:
                    continue

                # 垂直间隙
                gap = bot[1][:, 1].min() - top[1][:, 1].max()
                if not (-avg_h * 0.2 < gap < avg_h * 1.2):
                    continue

                text = ta + tb
                raw = _extract_plate(text)
                if not raw:
                    continue
                corrected, _ = correct(raw)
                if not is_valid(corrected):
                    continue

                groups.append([top, bot])
                taken.update((a[0], b[0]))
                break

        return groups

    def _merge_fragments(
        self, group: list[tuple], img: np.ndarray, scale: float
    ) -> PlateResult | None:
        """把同行分片拼成一个号码。"""
        text = "".join(_clean_text(f[2]) for f in group)
        # 同样走两级抽取：严格优先，纯字母数字兜底
        raw = _extract_plate(text)
        if not raw and not _BARE_TEXT.match(text):
            return None

        corrected, changes = correct(raw or text)
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
            layout=PlateLayout(measure_layout(roi)) if roi.size else PlateLayout.UNKNOWN,
            bbox=(x0, y0, x1 - x0, y1 - y0),
            roi=roi if roi.size else None,
            confidence=float(np.clip(0.5 * score + 0.5 * plate_score(corrected), 0, 1)),
            engine=self.name,
            char_confidences=[score] * len(corrected),
            meta={
                "ocr_backend": self._impl,
                "fragments": len(group),
                "corrections": changes,
                "path": "fragments",
            },
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


__all__ = [
    "DeepEngine",
    "split_double_line",
    "measure_layout",
    "_extract_plate",
    "_align_char_probs",
    "_weak_positions",
]
