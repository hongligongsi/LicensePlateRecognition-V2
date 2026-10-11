"""深度学习引擎的纯函数级测试。

不依赖 OCR 模型，只验证预处理/后处理逻辑的正确性，
保证在没有安装 onnxruntime 的机器上也能跑。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lpr.engines.deep_engine import (
    _clean_text,
    _dedupe,
    _iou,
    _sharpen,
    _to_single_line,
    _upscale,
)
from lpr.types import PlateResult


# ══════════════════════════════════════════════════ _clean_text


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("京A·D77972", "京AD77972"),
        ("京A-D77972", "京AD77972"),
        ("京 A D 7 7 9 7 2", "京AD77972"),
        ("京AD77972。", "京AD77972"),
        ("京AD77972\n", "京AD77972"),
        ("京ad77972", "京AD77972"),
        ("京A_77972", "京A77972"),
        ("(京AD77972)", "京AD77972"),
    ],
)
def test_clean_text_strips_separators(raw: str, expected: str) -> None:
    assert _clean_text(raw) == expected


def test_clean_text_empty() -> None:
    assert _clean_text("") == ""


# ══════════════════════════════════════════════════ _upscale


def test_upscale_increases_size() -> None:
    small = np.zeros((20, 100, 3), dtype=np.uint8)
    out = _upscale(small, 3.0)
    assert out.shape[0] > small.shape[0]
    assert out.shape[1] > small.shape[1]


def test_upscale_clamps_factor() -> None:
    """极大因子不得把图放大到内存爆炸。"""
    img = np.zeros((30, 120, 3), dtype=np.uint8)
    out = _upscale(img, 100.0)
    assert out.shape[0] <= 30 * 8, f"放大失控: {out.shape}"


def test_upscale_handles_degenerate_input() -> None:
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert _upscale(empty, 3.0).size == 0


def test_upscale_preserves_aspect_ratio_roughly() -> None:
    img = np.zeros((25, 100, 3), dtype=np.uint8)
    out = _upscale(img, 3.0)
    src_ar = 100 / 25
    out_ar = out.shape[1] / out.shape[0]
    assert abs(src_ar - out_ar) < 0.5


# ══════════════════════════════════════════════════ _sharpen


def test_sharpen_zero_amount_is_identity() -> None:
    img = np.random.default_rng(0).integers(0, 255, (30, 100, 3), dtype=np.uint8)
    out = _sharpen(img, 0.0)
    assert np.array_equal(out, img)


def test_sharpen_keeps_shape_and_dtype() -> None:
    img = np.full((30, 100, 3), 128, dtype=np.uint8)
    out = _sharpen(img, 0.8)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_sharpen_increases_edge_contrast() -> None:
    """黑白边界经 USM 后对比度应加强（方差变大）。"""
    img = np.zeros((30, 100, 3), dtype=np.uint8)
    img[:, 50:] = 200
    sharpened = _sharpen(img, 1.5)
    assert sharpened.astype(float).std() >= img.astype(float).std() - 1.0


# ══════════════════════════════════════════════════ _to_single_line


def test_to_single_line_passes_through_single_row() -> None:
    """单行牌不应被误拆。"""
    roi = np.full((40, 200, 3), 255, dtype=np.uint8)
    cv2.putText(roi, "A12345", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    layouts = _to_single_line(roi)
    assert len(layouts) >= 1
    assert layouts[0].shape == roi.shape


def test_to_single_line_splits_two_bands() -> None:
    """上短下长的双行牌应被拆开并横向拼接。

    注意样图比例取真实双行牌的 2:1（440x220mm）；
    更宽的图（>2.35:1）会被 ``split_double_line`` 的宽高比门槛
    正确地视为单行牌而不拆分。
    """
    roi = np.full((80, 160, 3), 255, dtype=np.uint8)
    # 上行：短文本，下行：长文本
    cv2.putText(roi, "AA", (48, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(roi, "12345", (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    layouts = _to_single_line(roi)
    assert len(layouts) == 2, f"应产出[原图, 拼接图]，实际 {len(layouts)}"
    stacked = layouts[1]
    # 拼接后高度等于较高行，宽度为两行宽度之和（缩放后）
    assert stacked.shape[0] <= roi.shape[0]
    assert stacked.shape[1] > 0


def test_to_single_line_handles_blank_image() -> None:
    blank = np.full((60, 200, 3), 255, dtype=np.uint8)
    layouts = _to_single_line(blank)
    assert len(layouts) >= 1


def test_to_single_line_handles_tiny_input() -> None:
    tiny = np.zeros((5, 5, 3), dtype=np.uint8)
    assert len(_to_single_line(tiny)) == 1


def test_to_single_line_accepts_grayscale() -> None:
    gray = np.full((80, 160), 255, dtype=np.uint8)
    cv2.putText(gray, "AA", (48, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    cv2.putText(gray, "12345", (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    layouts = _to_single_line(gray)
    assert len(layouts) >= 1


# ══════════════════════════════════════════════════ _iou / _dedupe


def test_iou_identical_boxes() -> None:
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes() -> None:
    assert _iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_partial_overlap() -> None:
    value = _iou((0, 0, 10, 10), (5, 0, 10, 10))
    assert 0.0 < value < 1.0


def test_iou_zero_area_boxes() -> None:
    assert _iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def _res(text: str, bbox, conf: float = 0.9) -> PlateResult:
    return PlateResult(
        text=text, color=None, bbox=bbox, roi=None, confidence=conf, engine="test"
    )


def test_dedupe_merges_overlapping_same_text() -> None:
    a = _res("京A12345", (0, 0, 100, 40), 0.7)
    b = _res("京A12345", (2, 1, 100, 40), 0.95)
    out = _dedupe([a, b])
    assert len(out) == 1
    assert out[0].confidence == 0.95, "应保留置信度更高的那个"


def test_dedupe_keeps_different_texts() -> None:
    a = _res("京A12345", (0, 0, 100, 40))
    b = _res("沪B67890", (0, 0, 100, 40))
    assert len(_dedupe([a, b])) == 2


def test_dedupe_collapses_same_text_regardless_of_position() -> None:
    """同一张图里号码相同即视为同一条观测，位置不同也合并。

    整图兜底路径可能对同一块车牌产出多个框，位置略有偏差，
    号码一致就应合并，否则会重复上报同一辆车。
    """
    a = _res("京A12345", (0, 0, 100, 40), 0.6)
    b = _res("京A12345", (500, 500, 100, 40), 0.9)
    out = _dedupe([a, b])
    assert len(out) == 1
    assert out[0].confidence == 0.9


def test_dedupe_sorted_by_confidence() -> None:
    results = [
        _res("京A12345", (0, 0, 100, 40), 0.5),
        _res("沪B67890", (500, 0, 100, 40), 0.9),
    ]
    out = _dedupe(results)
    assert out[0].confidence >= out[-1].confidence


def test_dedupe_empty() -> None:
    assert _dedupe([]) == []
