"""字符分割：把车牌图切成单个字符。

沿用原项目的"水平/垂直投影 + 找波峰"思路（这个思路本身是对的），
但修掉以下问题：

1. ``find_waves`` 里循环变量 ``i`` 在循环外被引用（原版第 38 行），
   Python 3 作用域下虽然能跑但语义脆弱，这里显式处理。
2. 原版靠"波峰宽度累加到 max_wave_dis * 0.6"来合并汉字，阈值靠试；
   这里改为按**目标字符数 7**做动态切分：先切，切多了合并最窄的两个，
   切少了从最宽处分裂，直到得到 7 个（或放弃）。
3. 原版遇到 ``len(wave_peaks) <= 6`` 直接 ``continue`` 放弃整个候选，
   导致明明能识别的情况也失败。这里改为"尽力切到 7 个"。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: 中国大陆普通车牌（含新能源）的字符数
STANDARD_CHAR_COUNT = 7


@dataclass
class Segmentation:
    """分割结果。"""

    #: 每个字符的二值图（白字黑底）
    chars: list[np.ndarray]
    #: 车牌区域二值化后的整图，便于调试
    binary: np.ndarray


def find_waves(threshold: float, histogram: np.ndarray) -> list[tuple[int, int]]:
    """在投影直方图中找出所有"波峰"区间（即字符所在的列/行区间）。

    原版实现能工作但边界处理不严：histogram 为空时会 ``IndexError``，
    循环结束后引用 ``i`` 也有隐患。这里重写为纯 python 但更严谨。
    """
    if histogram is None or len(histogram) == 0:
        return []

    peaks: list[tuple[int, int]] = []
    is_peak = False
    start = 0

    for i, value in enumerate(histogram):
        if value >= threshold:
            if not is_peak:
                is_peak = True
                start = i
        else:
            if is_peak:
                is_peak = False
                if i - start > 2:
                    peaks.append((start, i))

    # 收尾：最后一个波峰延伸到末端
    if is_peak and len(histogram) - start > 4:
        peaks.append((start, len(histogram)))

    return peaks


def binarize_plate(roi: np.ndarray, color_is_dark_text: bool) -> np.ndarray:
    """把车牌 ROI 转成"白字黑底"的二值图。

    蓝牌是深色字浅色底，黄/绿牌相反。原版用 ``bitwise_not`` 实现反色，
    这里保留该逻辑，但用自适应阈值 + OTSU 双保险，光照不均时更稳。
    """
    if roi is None or roi.size == 0:
        return np.zeros((1, 1), np.uint8)

    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    if color_is_dark_text:
        # 蓝牌：浅底深字 → 反色后变成黑底白字
        gray = cv2.bitwise_not(gray)

    # OTSU 全局阈值作为主方案
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 如果 OTSU 结果前景太少（小于 8%），说明阈值选偏了，改用自适应
    fg_ratio = cv2.countNonZero(binary) / binary.size
    if fg_ratio < 0.08:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
        )

    return binary


def _trim_rows(binary: np.ndarray) -> np.ndarray:
    """按水平投影裁掉车牌上下多余的白边/黑边。"""
    if binary.size == 0:
        return binary

    row_sum = np.sum(binary, axis=1).astype(np.float64)
    if row_sum.size == 0:
        return binary

    threshold = (row_sum.min() + row_sum.mean()) / 2
    peaks = find_waves(threshold, row_sum)
    if not peaks:
        return binary

    # 取最宽的波峰当作车牌主体所在行区间
    start, end = max(peaks, key=lambda p: p[1] - p[0])
    trimmed = binary[start:end]
    return trimmed if trimmed.size else binary


def _merge_to_count(
    spans: list[tuple[int, int]], target: int, width: int
) -> list[tuple[int, int]]:
    """通过合并相邻区间，把区间数降到 ``target``。

    合并策略：每次合并间隙最小的两个相邻区间（汉字之间的缝通常比字符内部的缝宽）。
    """
    spans = list(spans)
    while len(spans) > target:
        if len(spans) < 2:
            break
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        idx = int(np.argmin(gaps))
        spans[idx] = (spans[idx][0], spans[idx + 1][1])
        del spans[idx + 1]
    return spans


def _split_to_count(
    spans: list[tuple[int, int]], target: int, binary: np.ndarray
) -> list[tuple[int, int]]:
    """通过切分最宽的区间，把区间数提到 ``target``。

    切分点选该区间内部投影值最小的列（字符之间的天然缝隙）。
    """
    spans = list(spans)
    guard = 0
    while len(spans) < target and guard < 20:
        guard += 1
        widths = [s[1] - s[0] for s in spans]
        idx = int(np.argmax(widths))
        start, end = spans[idx]
        if end - start < 6:
            break

        # 在中间 60% 范围内找投影最小的一列
        lo = start + max(1, (end - start) // 5)
        hi = end - max(1, (end - start) // 5)
        if hi <= lo:
            break
        col_sum = np.sum(binary[:, lo:hi], axis=0).astype(np.float64)
        if col_sum.size == 0:
            break
        cut = lo + int(np.argmin(col_sum))
        if cut <= start or cut >= end:
            break

        spans[idx : idx + 1] = [(start, cut), (cut, end)]

    return spans


def separate_characters(
    roi: np.ndarray,
    color_is_dark_text: bool,
    expected_count: int = STANDARD_CHAR_COUNT,
) -> Segmentation:
    """把车牌图分割成单个字符图。

    Args:
        roi: 车牌区域（BGR 或灰度）。
        color_is_dark_text: True 表示浅底深字（蓝牌），需要反色。
        expected_count: 期望字符数，普通车牌为 7。

    Returns:
        ``Segmentation``，``chars`` 为按从左到右顺序排列的二值字符图。
    """
    binary = binarize_plate(roi, color_is_dark_text)
    if binary.size == 0:
        return Segmentation(chars=[], binary=binary)

    # 1) 裁掉上下白边，只留字符行
    body = _trim_rows(binary)
    if body.size == 0:
        return Segmentation(chars=[], binary=binary)

    # 2) 去掉最上一行和最下一行，避免车牌边框被当成字符（原版的经验做法）
    if body.shape[0] > 4:
        body = body[1:-1]

    # 3) 垂直投影找字符列
    col_sum = np.sum(body, axis=0).astype(np.float64)
    if col_sum.size == 0:
        return Segmentation(chars=[], binary=binary)

    # 阈值取 (最小值 + 均值) / 5，比原版更宽松 —— 原版用 /5 是同一思路
    threshold = (col_sum.min() + col_sum.mean()) / 5
    spans = find_waves(threshold, col_sum)
    if not spans:
        return Segmentation(chars=[], binary=binary)

    # 4) 去掉最左侧/最右侧那种"贴着边界的窄条"，它们是车牌边框不是字符
    if len(spans) > expected_count:
        max_width = max(s[1] - s[0] for s in spans)
        if spans[0][0] == 0 and spans[0][1] - spans[0][0] < max_width / 3:
            spans.pop(0)
        if spans and spans[-1][1] >= col_sum.size - 1 and (
            spans[-1][1] - spans[-1][0] < max_width / 3
        ):
            spans.pop()

    # 5) 去除车牌中间的铆钉/分隔点：位于第 3 个位置且很窄
    if len(spans) > 4:
        widths = [s[1] - s[0] for s in spans]
        max_width = max(widths)
        mid = spans[2]
        if mid[1] - mid[0] < max_width / 3:
            patch = body[:, mid[0] : mid[1]]
            # 铆钉/分隔点是孤立小点，前景占比很低
            if patch.size and cv2.countNonZero(patch) / patch.size < 0.35:
                spans.pop(2)

    # 6) 对齐到期望字符数
    if len(spans) > expected_count:
        spans = _merge_to_count(spans, expected_count, body.shape[1])
    elif len(spans) < expected_count:
        spans = _split_to_count(spans, expected_count, body)

    # 7) 切图，并丢弃前景过少的碎片（原版把这类当成"铆钉"直接跳过）
    chars: list[np.ndarray] = []
    for start, end in spans:
        patch = body[:, start:end]
        if patch.size == 0:
            continue
        fg = cv2.countNonZero(patch) / patch.size
        if fg < 0.05:
            continue
        chars.append(patch)

    return Segmentation(chars=chars, binary=body)


__all__ = [
    "Segmentation",
    "separate_characters",
    "binarize_plate",
    "find_waves",
    "STANDARD_CHAR_COUNT",
]
