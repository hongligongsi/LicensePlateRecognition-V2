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

同一候选用不同超分倍数跑多次识别，按语法合法性加权投票；
再把所有变体的**原始读数做逐列对齐融合**（ROVER 风格），
能补回被多数变体稳定丢掉的字符（如细笔画字母 E）。
得益于单次识别仅 10~50ms，即便跑 3 个尺度也只要 100ms 级别。
"""

from __future__ import annotations

import difflib
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import PROJECT_ROOT, EngineConfig, LocateConfig
from ..plate_rules import correct, is_valid, loose_match, plate_score
from ..types import PlateColor, PlateLayout, PlateResult
from ..utils.image import ensure_bgr, imread, resize_max_side
from ..utils.logging_setup import get_logger
from ..utils.security import Limits, check_image
from .base import BaseEngine, EngineUnavailable
from .locate import _red_mask, detect_color, locate_plates, refine_by_color

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
    r"[A-Z][A-Z0-9]{3,6}(?:危(?:险品?)?|[警挂学港澳使领])?"
    r"|[军空海北沈兰济南广成][A-Z][A-Z0-9]{3,6}[警挂学危]?"
    r"|[A-HJ-NP-Z0-9]{6}使"
    r"|使[0-9]{4,6}"
    r"|[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新][0-9]{3}[使领]"
    r"|WJ[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新][0-9]{4}[A-HJ-NP-Z0-9]?"
    r"|WJ[0-9]{5,7}"
    r"|民航[A-G][0-9]{3,4}"
    # 农机号牌（省份 + 7 位纯数字，如 皖1149885）：双行绿牌上/下行拼合。
    # 与其它省份开头分支无冲突（那些第 2 位均为字母）。
    r"|[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新][0-9]{7}"
    # 军牌 2012 式（双字母 + 5 位数字）没有汉字锚点，必须放在**最后**：
    # 放在前面会把 WJ012345 抢着切成 WJ01234（{5} 先匹配满就停）。
    r"|[A-HJ-NP-Z]{2}[0-9]{5}"
)
#: 纯字母数字（可能是号码的一部分）
_BARE_TEXT = re.compile(r"^[A-Z0-9]{5,8}$")

#: 超分倍数（按顺序尝试，命中即停）
_UPSCALE_STEPS = (3.0, 2.0, 4.0)
#: 锐化强度
_SHARPEN_LEVELS = (0.0, 0.8)

#: 早停前必须跑满的最少轮数。
#: 单轮读数缺乏"跨变体一致性"证据——远距离黄牌实测第 1 轮读错、
#: 第 3 轮（2.0x）才读对，第 1 轮就停等于放弃纠错机会。
_MIN_STOP_ROUNDS = 2

#: 领手存在弱置信位时继续收集证据的轮数上限（再差也认输，控制耗时）。
_MAX_WEAK_ROUNDS = 4

#: 融合阶段：插入列字符的采纳权重阈值（相对参考串分数的比例）。
#: 低于它视为孤证噪声，不补进融合串。
_FUSED_INSERT_RATIO = 0.5

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
#: 取 0.50 而非更高：**农机双层牌上行（省简称+2 位机关码）字号本来就比
#: 下行小一号**（实测 double_lv.png 两行带高比 0.53），"行高必须相近"
#: 对农机牌不成立；而单行牌的装饰条厚度通常不足字符行的 1/3，0.50 仍可拦截。
_BAND_HEIGHT_SIMILARITY = 0.50

#: **红色尾字**集合：警牌末位「警」、领馆牌末位「领」是红色字符。
#: 通用 OCR 对"小字号 + 低对比"的红色字符经常整字漏读（README 已知限制），
#: 实测 沪A1707领 稳定读成 沪A1707、鄂F8888警 可能读成 鄂F8888。
#: 只有这两个尾字是红色，其余后缀（学/挂/危/港/澳）为黑色，不适用此路径。
_RED_TAIL_CHARS = frozenset("警领")


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


def _accumulate_char_probs(store: dict[str, list[float]], text: str, probs: list[float]) -> None:
    """把一次识别的逐字符置信度并入累积表（逐位取最大值）。"""
    if not text or not probs:
        return
    existing = store.get(text)
    if existing is None or len(existing) != len(probs):
        store[text] = list(probs)
        return
    store[text] = [max(a, b) for a, b in zip(existing, probs, strict=False)]


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


def _fuse_reads(reads: list[tuple[str, float]]) -> str | None:
    """把同一候选区的多次 OCR 读数做**逐列对齐融合**（ROVER 风格）。

    多尺度/多预处理变体读的是**同一块车牌**，但整串投票把每次读数当作
    独立字符串计票，会输给"同一个错读重复多次"——实测远距离黄牌
    ``京E51619``：5 个变体把细笔画的 E 读丢（``京51619``），仅 1 个变体
    读对，整串投票永远输。融合把所有读数按字符位置对齐后逐列投票：

    * **多数列**：取加权多数字符（参考串自身也参与投票）；
    * **插入列**（参考串缺失、其他读数在该位置稳定多出的字符）：
      权重达到 ``_FUSED_INSERT_RATIO × 参考串分数`` 才补回——
      既能捞回被丢掉的字母，又不给孤证噪声开门；
    * **删除列**（参考串有、其他读数没有）：不删——OCR 丢字远比多字
      常见，且删字交给调用方的 ``_extract_plate`` 闸门兜底。

    Returns:
        融合串。读数为空返回 ``None``，只有一条读数时原样返回。
        坏融合（对齐错位、噪声混入）由调用方用 ``_extract_plate`` 滤掉。
    """
    if not reads:
        return None
    ref, ref_score = max(reads, key=lambda r: (r[1], len(r[0])))
    if len(reads) == 1:
        return ref

    # 逐列加权票：cols[i] 是参考串第 i 个字符位置的 {字符: 票重}
    cols: list[dict[str, float]] = [{ch: ref_score} for ch in ref]
    # 插入列：ins[pos] 表示"插在参考串位置 pos 之前"的 {字符: 票重}
    ins: dict[int, dict[str, float]] = {}

    for text, score in reads:
        if text == ref:
            continue
        sm = difflib.SequenceMatcher(a=ref, b=text, autojunk=False)
        for op, a1, a2, b1, b2 in sm.get_opcodes():
            if op == "equal":
                # 与参考串一致：给参考字符追加票重（多数列的"同意票"）
                for k in range(a2 - a1):
                    ch = ref[a1 + k]
                    cols[a1 + k][ch] = cols[a1 + k].get(ch, 0.0) + score
            elif op == "replace":
                # 等长部分按列投票；不等长时多出的当插入、缺的当删除
                n = min(a2 - a1, b2 - b1)
                for k in range(n):
                    ch = text[b1 + k]
                    cols[a1 + k][ch] = cols[a1 + k].get(ch, 0.0) + score
                if b2 - b1 > n:
                    bucket = ins.setdefault(a2, {})
                    for k in range(b1 + n, b2):
                        bucket[text[k]] = bucket.get(text[k], 0.0) + score
                # a2-a1 > n 的部分是对方丢字：不投反对票
            elif op == "insert":
                bucket = ins.setdefault(a1, {})
                for k in range(b1, b2):
                    bucket[text[k]] = bucket.get(text[k], 0.0) + score
            # delete：对方丢了参考串的字符，OCR 丢字常见，不投反对票

    fused = list(ref)

    # 插入列：权重过阈值才补回（从后往前插，避免下标失效）
    for pos in sorted(ins, reverse=True):
        ch, weight = max(ins[pos].items(), key=lambda kv: kv[1])
        if weight >= _FUSED_INSERT_RATIO * ref_score:
            fused.insert(pos, ch)

    # 多数列：挑战者总票重明显压过参考字符才替换（保守，防止多数被带偏）
    for i, tally in enumerate(cols):
        ch, weight = max(tally.items(), key=lambda kv: kv[1])
        if ch != ref[i] and weight > tally.get(ref[i], 0.0):
            fused[i] = ch

    return "".join(fused)


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
        new_w = max(1, round(band.shape[1] * scale))
        return cv2.resize(band, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    stacked = np.hstack([_fit_height(_trim_band_cols(top)), _fit_height(_trim_band_cols(bottom))])
    # 拼接后高宽比若已达单行牌特征（≥2.5:1），说明两行都是有效字符行
    return [roi, stacked]


def _trim_band_cols(band: np.ndarray) -> np.ndarray:
    """裁掉行带左右两侧的固封螺丝/牌框竖边等非字符列。

    双层牌上行的固封螺丝是个圆形小铆钉，拼接进单行形态后会被 OCR
    读成 "0"/"8"（实测 double_lv.png 上行"皖11"被读成"院078"）。
    螺丝的列前景数不超过自身直径（远小于行高），而字符笔画纵贯行带，
    用"列前景数 > 行高一半"即可只保留字符列；取最左/最右活跃列裁剪，
    中间的间隙列（字符间空隙）原样保留。
    """
    h = band.shape[0]
    binary = _char_binary(band)
    col_counts = (binary > 0).sum(axis=0)
    active = np.nonzero(col_counts > max(2, int(0.5 * h)))[0]
    if active.size < 2:
        return band
    pad = max(1, int(0.05 * h))
    x0 = max(0, int(active[0]) - pad)
    x1 = min(band.shape[1], int(active[-1]) + 1 + pad)
    return band[:, x0:x1] if x1 - x0 >= 4 else band


def _band_profile(roi: np.ndarray) -> np.ndarray:
    """算出行方向的前景像素投影，用于定位字符行带。

    关键细节：**先裁掉左右各 8% 的边框**。车牌四周有 1~2px 的白色/黑色
    框线，它在每一行都会贡献前景像素，会把上下两行字符"桥接"成一段连续
    行带，导致永远切不出两段。

    （参考实现 we0091234/YOLO26-plate 直接按固定比例 5/12、1/3 切分来
    绕开这个问题；我们改成"裁边框 + 投影"的通用做法，不依赖固定比例，
    对各种宽高比的输入都成立。）
    """
    _h, w = roi.shape[:2]
    margin = round(w * _BORDER_TRIM)
    if w - 2 * margin >= 0.4 * w:
        core = roi[:, margin : w - margin]
    else:
        core = roi

    return (_char_binary(core) > 0).sum(axis=1)


def _char_binary(roi: np.ndarray) -> np.ndarray:
    """极性自适应的字符二值图（前景=字符）。

    字符在牌面上永远是**少数派**（约占 20%~40% 面积）。
    黄/绿牌是深字浅底，INV 极性前景稀疏（正确）；而绿底白字、黑底白字的
    双层牌用 INV 会把**底色**当成前景——每行都铺满前景，行带永远切不出来
    （实测 double_lv.png 的双层绿牌即卡在这里）。哪种极性的前景更稀疏，
    哪种才是字符。退化情形（前景占比 <2%，基本是噪声）维持 INV 旧行为。
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inv_fg = float((inv > 0).mean())
    if inv_fg < 0.02:
        return inv
    _, fwd = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fwd_fg = float((fwd > 0).mean())
    return fwd if fwd_fg < inv_fg else inv


def _find_bands(profile: np.ndarray, h: int, width: int) -> list[tuple[int, int]]:
    """从投影里找出字符行带，并容忍被边框桥接的小缺口。

    单行牌字符高度可达车牌高度的 60% 以上，若某段行带占比超过
    ``_SINGLE_BAND_RATIO``，说明它已经是一整行字符（或两行被桥接在一起），
    此时不再往下切。这是"带边框单行牌被误拆"的最后一道防线。
    """
    if profile.size == 0 or profile.max() <= 0:
        return []

    # 活跃阈值取宽度的 8%：字符行的前景量级是每行几十像素，而双行牌
    # 行间隙里常有 5% 左右的纹理/反光残噪（实测 double_lv.png 间隙行
    # 仍有 6~7px/行），2% 的阈值会把间隙误判为活跃、把两行桥成一行。
    active = profile > max(2, int(0.08 * width))
    # 桥接：把宽度小于 min_gap 的空白当噪声填掉（框线的抗锯齿/断开）
    min_gap = max(1, round(h * _BAND_BRIDGE_RATIO))
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

    # 边缘细条过滤：紧贴 ROI 上/下缘的薄带是**车牌水平框线**（白色牌框在
    # 投影顶部/底部各形成一条几像素高的薄带），不是字符行。不滤掉的话
    # 双行牌会被数出 3 段行带（框线+两行字），永远拆不开。
    n = len(profile)
    result = [b for b in result if not ((b[1] - b[0]) < 0.10 * h and (b[0] == 0 or b[1] == n))]
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
    bgr = ensure_bgr(roi, min_side=12)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]

    # 宽高比过大 → 不可能是双行牌（双行牌高度约为宽度的一半）
    if w / max(1, h) > _DOUBLE_MAX_ASPECT:
        return None

    profile = _band_profile(bgr)
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

    top = bgr[bands[0][0] : bands[0][1], :]
    bottom = bgr[bands[1][0] : bands[1][1], :]
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

    bgr = ensure_bgr(roi, min_side=8)
    if bgr is None:
        return "unknown"

    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return "unknown"

    # ---- 证据 1（主导）：行分割
    # 它能直接切出两条高度相近、上下分离的行带 —— 这是双行牌最硬的证据。
    if split_double_line(bgr) is not None:
        return "double"

    # 行分割"明确否定"：出现了内容过高（超过 50%）的单条行带。
    # 那事实上就是一行字符，比宽高比先验可靠得多，直接判单行。
    if _has_single_band(bgr):
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
    bgr = ensure_bgr(roi, min_side=12)
    if bgr is None:
        return False
    h, w = bgr.shape[:2]
    profile = _band_profile(bgr)
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

    est_char_h = h * 0.62  # 字符高约占车牌高 62%
    ideal = 48.0 / max(1.0, est_char_h)  # 达到 48px 字符高所需的倍数
    factor = float(np.clip((ideal + factor) / 2.0, 1.0, 6.0))

    new_w, new_h = max(1, int(w * factor)), max(1, int(h * factor))
    if new_w * new_h > 2_000_000:  # 内存保护
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
        self._ocr: Any = None
        self._recognizer = None
        #: 第二识别器（低置信交叉验证），仅 rapidocr 后端 + 配置启用时加载
        self._recognizer2 = None
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
            self._rec_only_supported = self._recognizer is not None and hasattr(
                self._ocr, "get_boxes_img_without_det"
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

        # 可选：加载第二识别模型（低置信交叉验证）。任何失败都不影响主引擎。
        if self._impl == "rapidocr" and self.config.secondary_rec_model:
            self._load_secondary_recognizer()

        logger.info(
            "深度学习引擎就绪 | 后端=%s | 定位引导=%s | 纯识别加速=%s | 第二模型=%s",
            self._impl,
            "启用" if self._locator_ready else "关闭",
            "启用" if self._rec_only_supported else "关闭",
            "启用" if self._recognizer2 is not None else "关闭",
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

    def _load_secondary_recognizer(self) -> None:
        """加载第二识别模型（低置信交叉验证）。

        只取其中的**纯识别器**（text_recognizer），不跑它的检测网络。
        任何异常都吞掉并记日志——交叉验证是锦上添花，绝不能让
        第二模型的加载问题拖垮主引擎。

        注意：rapidocr-onnxruntime 1.2.3 的 UpdateParameters 有缺陷，
        ``rec_model_path`` 不能与任何 ``det_*`` 参数同传（会 KeyError），
        因此这里只传模型路径。
        """
        try:
            path = Path(self.config.secondary_rec_model)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.exists():
                logger.warning("第二识别模型不存在: %s，交叉验证关闭", path)
                return

            from rapidocr_onnxruntime import RapidOCR

            ocr2 = RapidOCR(rec_model_path=str(path))
            self._recognizer2 = getattr(ocr2, "text_recognizer", None)
            if self._recognizer2 is None:
                logger.warning("第二识别模型无 text_recognizer，交叉验证关闭")
        except Exception:
            logger.warning("第二识别模型加载失败，交叉验证关闭", exc_info=True)
            self._recognizer2 = None

    def _recognize2_detailed(self, img: np.ndarray) -> list[tuple[str, float, list[float]]]:
        """用第二识别模型读一张图，输出格式与 :meth:`_recognize_only_detailed` 一致。

        主路径的 ``get_boxes_img_without_det`` 会把整图视作一个文本框再切块，
        这里直接送 ``[img]`` 给识别器即可（调用方传入的本来就是裁好的车牌）。
        """
        if self._recognizer2 is None:
            return []
        try:
            results, _ = self._recognizer2([img])
        except Exception:
            logger.debug("第二模型识别异常", exc_info=True)
            return []

        out: list[tuple[str, float, list[float]]] = []
        for item in results or []:
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue
            text, score = item[0], item[1]
            if not text or float(score) <= 1e-3:
                continue
            char_probs = self._extract_char_probs(item, str(text), float(score))
            out.append((str(text), float(score), char_probs))
        return out

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

        logger.debug("深度引擎 %.1f ms 命中 %d 个车牌 (%s)", elapsed, len(results), path_used)
        return results

    # -------------------------------------------------------------- 候选

    def _guided_candidates(self, img: np.ndarray) -> list[tuple[np.ndarray, PlateColor, tuple]]:
        """用 CV 定位模块圈出车牌候选区。"""
        if not self._locator_ready:
            return []

        h, w = img.shape[:2]
        if max(h, w) < 200:  # 已经是车牌特写，无需定位
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

    def _recognize_only_detailed(self, img: np.ndarray) -> list[tuple[str, float, list[float]]]:
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

    def _read_red_tail(self, roi: np.ndarray) -> tuple[str, float] | None:
        """对车牌最右侧的红色字符条带做定向识别，返回 ``(字符, 置信度)``。

        背景：警牌末位「警」、领馆牌末位「领」是**红色**字符，通用 OCR
        模型对这种"小字号 + 低对比 + 非常规颜色"的字符经常整字漏读
        （README 已知限制）。但红色本身是强特征——先用红色掩码列投影
        精确定位它在车牌上的横向位置，把这一条带裁出来放大单独识别，
        就把"全文找漏字"降维成了"单字认读"，识别网络只需处理一个字符。

        Returns:
            ``(红色尾字, 置信度)``；定位失败或认读不出合法尾字返回 ``None``。
        """
        if roi is None or roi.size == 0 or roi.ndim == 2:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = _red_mask(hsv)
        h, w = roi.shape[:2]
        # 红像素总量过小 → 没有红色字符（挡掉零星反光）
        if cv2.countNonZero(mask) < mask.size * 0.004:
            return None

        # 列投影找红色字符簇，取**最右侧**的一簇（警/领都在号码末位）
        col_hits = cv2.reduce(mask, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() / 255.0
        active = col_hits > max(1.0, 0.06 * h)
        clusters: list[list[int]] = []
        start = None
        for i, flag in enumerate(active):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                clusters.append([start, i - 1])
                start = None
        if start is not None:
            clusters.append([start, len(active) - 1])
        # 合并被 1~2px 噪声断开的相邻簇
        merged: list[list[int]] = []
        for c in clusters:
            if merged and c[0] - merged[-1][1] <= 2:
                merged[-1][1] = c[1]
            else:
                merged.append(c)
        if not merged:
            return None
        x0, x1 = merged[-1]
        width = x1 - x0 + 1
        # 单字符的合理宽度：车牌字符宽约为字高的 0.4~0.8 倍，条带高度≈车牌高
        if not (0.15 * h <= width <= 1.6 * h):
            return None
        # 簇必须靠近右缘（尾字）；左侧红字是军牌字头/武警 WJ，不走此路径
        if x1 < 0.65 * w:
            return None

        pad = max(2, round(width * 0.25))
        band = roi[:, max(0, x0 - pad) : min(w, x1 + pad)]

        # 1) 窄条带单字认读：图像质量够时 OCR 能直接读出该字
        for factor in (5.0, 8.0, 3.0):
            up = cv2.resize(band, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            for text, score in self._recognize_only(up):
                cleaned = _clean_text(text)
                if len(cleaned) == 1 and cleaned in _RED_TAIL_CHARS:
                    return cleaned, float(score)

        # 2) 宽条带上 下文认读：窄裁剪信息量不足时，带上左侧邻字识别，
        #    取读数**末位**。实测低质领馆牌会把「领」读成 0/7 等形近字符，
        #    认不出时交给底色推断兜底。
        wide = roi[:, max(0, x0 - int(2.5 * width)) : min(w, x1 + pad)]
        for factor in (4.0, 6.0):
            up = cv2.resize(wide, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            for text, score, _probs in self._recognize_only_detailed(up):
                cleaned = _clean_text(text)
                if cleaned and cleaned[-1] in _RED_TAIL_CHARS:
                    return cleaned[-1], float(score)
        return None

    @staticmethod
    def _infer_red_tail(roi: np.ndarray) -> tuple[str, float] | None:
        """按**底色极性**推断红色尾字的身份，返回 ``(字符, 置信度)``。

        依据号牌板式的法定用色（GA36）：红色尾字只有两种——
        警用号牌是**白底红「警」**，使领馆号牌是**黑底红「领」**。
        因此只要确认"存在红色尾字 + 底色极性"，字符身份就是确定的：
        这在字符太糊、OCR 认读失败时（实测会把「领」读成 0）仍能成立。

        仅当底色是**纯白或纯黑**（而非蓝/黄/绿底混入白字）时才推断，
        配合调用方的语法分闸门，误补面被双重兜住。
        """
        if roi is None or roi.size == 0 or roi.ndim == 2:
            return None
        color, ratio = detect_color(roi, LocateConfig())
        if color is PlateColor.BLACK and ratio >= 0.3:
            return "领", min(0.8, 0.4 + ratio * 0.5)
        if color is PlateColor.WHITE and ratio >= 0.3:
            return "警", min(0.8, 0.4 + ratio * 0.5)
        return None

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
            #: 各变体的**原始** OCR 读数（清洗后、未抽取），供逐列融合使用
            raw_reads: list[tuple[str, float]] = []
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
                        if not cleaned:
                            continue
                        raw_reads.append((cleaned, score))
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
                    # 早停三道闸门（全部满足才停）：
                    #   1. 至少 _MIN_STOP_ROUNDS 轮 —— 单轮读数没有跨变体一致性；
                    #   2. 领手是 7/8 位合法车牌 —— 6 位"短牌"读数常是丢字后被
                    #      纠错规则"修"出来的假合法串（实测 京E51619→京S1619），
                    #      必须跑满计划给正确读数出场机会；
                    #   3. 票份过半。领手存在弱置信位时继续收集证据（封顶限轮）。
                    if (
                        rounds >= _MIN_STOP_ROUNDS
                        and len(leader) in (7, 8)
                        and is_valid(leader)
                        and share >= 0.5
                    ):
                        probs = char_probs_by_text.get(leader) or []
                        if not _weak_positions(probs) or rounds >= _MAX_WEAK_ROUNDS:
                            logger.debug("候选收敛于 %s（%d/%d 轮）", leader, rounds, len(plan))
                            break

            # ---- 低置信交叉验证：主模型对该候选完全无产出时，
            # 用第二识别模型补读。只会在"本来就要丢弃"的候选上触发，
            # 对已命中的候选零影响；第二模型票权略低于主模型，
            # 靠错误模式互补把失败样本救回来。
            secondary_used = False
            if not votes and self._recognizer2 is not None:
                weight = float(np.clip(self.config.secondary_vote_weight, 0.0, 1.0))
                for up, sharp in plan:
                    for variant in variants:
                        prepared = _sharpen(_upscale(variant, up), sharp)
                        for text, score, char_probs in self._recognize2_detailed(prepared):
                            cleaned = _clean_text(text)
                            if not cleaned:
                                continue
                            raw_reads.append((cleaned, score * weight))
                            candidate = _extract_plate(cleaned)
                            if not candidate:
                                continue
                            secondary_used = True
                            votes[candidate] = votes.get(candidate, 0.0) + score * weight * (
                                0.5 + plate_score(candidate)
                            )
                            _accumulate_char_probs(char_probs_by_text, candidate, char_probs)
                    # 已救回读数即可停：补救路径只求"有"，不求穷尽所有尺度
                    if votes:
                        break

            if not votes:
                continue

            # ---- 读数融合（ROVER 风格）：跨变体逐列共识
            # 整串投票会被"同一错读重复多轮"带偏（每轮错读都单独计票）；
            # 逐列融合把所有变体的读数对齐投票，能补回被稳定丢掉的字符。
            fused_cand: str | None = None
            if len(raw_reads) >= 2:
                fused_text = _fuse_reads(raw_reads)
                if fused_text:
                    cand = _extract_plate(fused_text)
                    if cand:
                        fused_cand = cand
                        ref_score = max(s for _, s in raw_reads)
                        votes[cand] = votes.get(cand, 0.0) + ref_score * (0.5 + plate_score(cand))

            winner = max(votes, key=lambda k: votes[k])
            # 融合共识与整串投票领手不一致时，结构分更高者胜：
            # 逐列融合代表所有变体的真实共识，优先级高于整串票数。
            if (
                fused_cand is not None
                and fused_cand != winner
                and plate_score(correct(fused_cand)[0]) > plate_score(correct(winner)[0])
            ):
                logger.debug("融合读数 %s 压过投票领手 %s", fused_cand, winner)
                winner = fused_cand

            share = votes[winner] / (sum(votes.values()) or 1.0)
            share = max(share, 1.0 / max(1, len(votes)))

            corrected, changes = correct(winner)

            # ---- 红色尾字定向识别（警/领）：OCR 对小字号红色字符常整字漏读，
            # 这里用红色信号定位末位条带单独认读；认读失败时按底色极性推断
            # （白底红尾字必是警、黑底红尾字必是领，GA36 法定用色）。
            # 只在号码当前无中文尾字、长度合理、且补上后语法分**严格更高**时
            # 采纳（防止把领馆牌错补成警牌——两者补后都合法，同分的不动）。
            if corrected and corrected[-1].isascii() and len(corrected) in (5, 6, 7):
                red_tail = self._read_red_tail(roi) or self._infer_red_tail(roi)
                if red_tail is not None and plate_score(corrected + red_tail[0]) > plate_score(
                    corrected
                ):
                    changes.append(f"红色尾字定向识别补回 {red_tail[0]!r}")
                    logger.debug("红字定向识别: %s → %s", corrected, corrected + red_tail[0])
                    corrected += red_tail[0]
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
                        # 该结果是否由第二模型交叉验证救回
                        "secondary_rescue": secondary_used,
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
                (np.asarray(item[0], np.float32), str(item[1]), float(item[2])) for item in result
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

    def _group_double_fragments(self, boxes: list[tuple], used: set[int]) -> list[list[tuple]]:
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
    "_align_char_probs",
    "_extract_plate",
    "_fuse_reads",
    "_weak_positions",
    "measure_layout",
    "split_double_line",
]
