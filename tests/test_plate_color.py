"""车牌底色与危险品运输牌识别测试。

覆盖用户需求「支持五种颜色：黑色/蓝色/绿色/白色/黄色，支持危险品」。

关键在于：危险品牌（黑底黄字）与普通黄牌（黄底黑字）**底色相反**，
必须靠字符/底色的明暗极性关系区分，不能只看"有没有黄色"。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lpr.config import LocateConfig
from lpr.engines.locate import detect_color, detect_hazmat
from lpr.types import COLOR_ALIASES, COLOR_NAMES, COLOR_SWATCH, PlateColor, parse_color


def _plate(bg: tuple[int, int, int], fg: tuple[int, int, int], text: str = "A1234") -> np.ndarray:
    """合成一块车牌：bg 为底色、fg 为字色（均为 BGR）。"""
    img = np.full((60, 220, 3), bg, np.uint8)
    if text:
        cv2.putText(img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, fg, 3)
    return img


# ══════════════════════════════════════════════ 枚举与常量


def test_five_color_names_defined() -> None:
    assert set(COLOR_NAMES) == {"黑色", "蓝色", "绿色", "白色", "黄色"}
    assert len(COLOR_NAMES) == 5


def test_enum_has_five_colors_plus_hazmat() -> None:
    values = {c.value for c in PlateColor}
    for expected in ("black", "blue", "green", "white", "yellow", "hazmat"):
        assert expected in values


def test_every_color_has_label_and_swatch() -> None:
    for color in PlateColor:
        assert color.label, f"{color} 缺少 label"
        assert color in COLOR_SWATCH, f"{color} 缺少色块"


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("黑色", PlateColor.BLACK),
        ("黑", PlateColor.BLACK),
        ("black", PlateColor.BLACK),
        ("蓝色", PlateColor.BLUE),
        ("蓝", PlateColor.BLUE),
        ("blue", PlateColor.BLUE),
        ("绿色", PlateColor.GREEN),
        ("绿", PlateColor.GREEN),
        ("green", PlateColor.GREEN),
        ("白色", PlateColor.WHITE),
        ("白", PlateColor.WHITE),
        ("white", PlateColor.WHITE),
        ("黄色", PlateColor.YELLOW),
        ("黄", PlateColor.YELLOW),
        ("yellow", PlateColor.YELLOW),
        ("危险品", PlateColor.HAZMAT),
        ("危险品运输", PlateColor.HAZMAT),
    ],
)
def test_parse_color_accepts_chinese_and_english(alias: str, expected: PlateColor) -> None:
    assert parse_color(alias) == expected


def test_parse_color_strips_whitespace_and_case() -> None:
    assert parse_color("  蓝色  ") == PlateColor.BLUE
    assert parse_color("YELLOW") == PlateColor.YELLOW


def test_parse_color_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="未知车牌颜色"):
        parse_color("紫色")


def test_color_aliases_only_reference_valid_colors() -> None:
    for value in COLOR_ALIASES.values():
        assert isinstance(value, PlateColor)


# ══════════════════════════════════════════════ 五种底色判定


@pytest.mark.parametrize(
    "name,bg,fg,expected",
    [
        ("蓝底白字", (200, 80, 20), (255, 255, 255), PlateColor.BLUE),
        ("黄底黑字", (0, 220, 255), (10, 10, 10), PlateColor.YELLOW),
        ("绿底黑字", (80, 200, 60), (10, 10, 10), PlateColor.GREEN),
        ("白底黑字", (245, 245, 245), (10, 10, 10), PlateColor.WHITE),
        ("黑底白字", (10, 10, 10), (255, 255, 255), PlateColor.BLACK),
    ],
)
def test_detect_five_colors(
    name: str, bg: tuple[int, int, int], fg: tuple[int, int, int], expected: PlateColor
) -> None:
    got, ratio = detect_color(_plate(bg, fg), LocateConfig())
    assert got is expected, f"{name} 判成 {got.label}（ratio={ratio:.2f}）"


# ══════════════════════════════════════════════ 危险品牌


def test_hazmat_is_black_bg_yellow_text() -> None:
    got, _ = detect_color(_plate((10, 10, 10), (0, 220, 255)), LocateConfig())
    assert got is PlateColor.HAZMAT


def test_normal_yellow_plate_is_not_hazmat() -> None:
    """黄底黑字是普通黄牌，绝不能被判成危险品。"""
    got, _ = detect_color(_plate((0, 220, 255), (10, 10, 10)), LocateConfig())
    assert got is PlateColor.YELLOW


def test_black_plate_with_white_text_is_not_hazmat() -> None:
    """黑底白字是普通黑牌，不是危险品牌。"""
    got, _ = detect_color(_plate((10, 10, 10), (255, 255, 255)), LocateConfig())
    assert got is PlateColor.BLACK


def test_detect_hazmat_true_for_black_on_yellow() -> None:
    assert detect_hazmat(_plate((10, 10, 10), (0, 220, 255)), LocateConfig()) is True


def test_detect_hazmat_false_for_yellow_on_black() -> None:
    assert detect_hazmat(_plate((0, 220, 255), (10, 10, 10)), LocateConfig()) is False


def _gradient_taxi_plate() -> np.ndarray:
    """合成黄绿渐变出租车牌：黄头（省份简称区）+ 绿身（号码区）。

    实拍样本 ``img/ef55d67fd01616bafe1d540500b512a2.jpg``（浙B·08520D）的
    整框实测：yellow≈0.17、dark≈0.49、green≈0.43 —— 前三条危险品判据
    （黑底够多 + 黄色前景存在 + 白色很少）全部满足，v2.8.2 及以前
    误判成危险品。合成牌比例相近（yellow≈0.20 / dark≈0.80 / green≈0.66）。
    """
    img = np.full((50, 127, 3), (25, 60, 25), np.uint8)  # 深绿身（暗、偏绿）
    img[:, :26] = (30, 200, 230)  # 黄头（渐变牌的省份简称区）
    cv2.putText(img, "08520D", (32, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    return img


def test_gradient_taxi_plate_is_not_hazmat() -> None:
    """黄绿渐变出租车牌不是危险品：绿色护栏必须拦住它。"""
    roi = _gradient_taxi_plate()
    assert detect_hazmat(roi, LocateConfig()) is False


def test_gradient_taxi_plate_detects_green() -> None:
    """渐变牌应判绿（绿身占大头），而非危险品或黄牌。"""
    got, _ = detect_color(_gradient_taxi_plate(), LocateConfig())
    assert got is PlateColor.GREEN


def test_hazmat_green_guard_configurable() -> None:
    """护栏可通过配置关闭（调大上限 / 调大倍数 → 恢复旧的误判行为）。"""
    roi = _gradient_taxi_plate()
    cfg = LocateConfig(hazmat_max_green=0.99, hazmat_green_yellow_ratio=99.0)
    assert detect_hazmat(roi, cfg) is True


def test_detect_hazmat_false_for_plain_black() -> None:
    """纯黑无文字不应判为危险品（缺少黄色前景）。"""
    assert detect_hazmat(_plate((10, 10, 10), (10, 10, 10), text=""), LocateConfig()) is False


def test_detect_hazmat_disabled_by_config() -> None:
    cfg = LocateConfig(detect_hazmat=False)
    assert detect_hazmat(_plate((10, 10, 10), (0, 220, 255)), cfg) is False
    # 关闭后黑底黄字会退化为黑牌（因为黑底占比更高）
    got, _ = detect_color(_plate((10, 10, 10), (0, 220, 255)), cfg)
    assert got is PlateColor.BLACK


def test_detect_hazmat_handles_empty_and_grayscale() -> None:
    cfg = LocateConfig()
    assert detect_hazmat(None, cfg) is False
    assert detect_hazmat(np.empty((0, 0, 3), np.uint8), cfg) is False

    # 灰度图丢失色相，无法区分"黑底黄字"与"黑底白字"，应保守返回 False
    # （宁可漏判为黑牌，也不要凭亮度猜出错误的"危险品"结论）
    gray = cv2.cvtColor(_plate((10, 10, 10), (0, 220, 255)), cv2.COLOR_BGR2GRAY)
    assert detect_hazmat(gray, cfg) is False


def test_detect_color_handles_empty_input() -> None:
    color, ratio = detect_color(None, LocateConfig())
    assert color is PlateColor.UNKNOWN
    assert ratio == 0.0


def test_hazmat_label_mentions_polarity() -> None:
    """标签应能提示用户这是黑底黄字，避免与黄牌混淆。"""
    label = PlateColor.HAZMAT.label
    assert "危险品" in label
    assert "黑底黄字" in label


# ══════════════════════════════════════════════ 危险品号码格式


@pytest.mark.parametrize("text", ["京A1234危", "京A12345危", "粤B1234危"])
def test_hazmat_number_format_recognized(text: str) -> None:
    from lpr.plate_rules import describe, is_valid

    assert is_valid(text), f"{text} 应合法"
    info = describe(text)
    assert info["type"] == "危险品运输"
    assert info["is_hazmat"] is True


def test_hazmat_flag_false_for_normal_plate() -> None:
    from lpr.plate_rules import describe

    assert describe("京A12345")["is_hazmat"] is False
    assert describe("京AD77972")["is_hazmat"] is False


def test_hazmat_extracted_by_ocr_regex() -> None:
    """OCR 文本抽取正则必须能保留末位「危」，否则号码会被截断。"""
    from lpr.engines.deep_engine import _PLATE_TEXT

    match = _PLATE_TEXT.search("京A1234危")
    assert match is not None
    assert match.group(0) == "京A1234危"


def test_other_suffix_plates_still_extracted() -> None:
    from lpr.engines.deep_engine import _PLATE_TEXT

    for raw, expected in [
        ("京A1234警", "京A1234警"),
        ("京A1234挂", "京A1234挂"),
        ("京A123学", "京A123学"),
        ("粤Z1234港", "粤Z1234港"),
    ]:
        match = _PLATE_TEXT.search(raw)
        assert match is not None, f"{raw} 未被抽取"
        assert match.group(0) == expected, f"{raw} 抽成 {match.group(0)}"


# ══════════════════════════════════════════════ BatchOutcome 底色过滤


def _plate_result(color: PlateColor, text: str = "京A12345"):
    from lpr.types import PlateResult

    return PlateResult(text=text, color=color, bbox=(0, 0, 100, 40), confidence=0.9)


def test_batch_keep_colors_filters_by_plate_color() -> None:
    from lpr.api import BatchOutcome, RecognizeOutcome

    batch = BatchOutcome(
        outcomes=[
            RecognizeOutcome(source="a.jpg", results=[_plate_result(PlateColor.YELLOW)]),
            RecognizeOutcome(source="b.jpg", results=[_plate_result(PlateColor.HAZMAT)]),
            RecognizeOutcome(source="c.jpg", results=[_plate_result(PlateColor.BLUE)]),
        ],
        started=0.0,
    )
    batch.keep_colors({PlateColor.HAZMAT})
    assert batch.hit == 1
    assert batch.all_texts == ["京A12345"]
    assert batch.outcomes[0].results == []
    assert len(batch.outcomes[1].results) == 1


def test_batch_keep_colors_accepts_multiple() -> None:
    from lpr.api import BatchOutcome, RecognizeOutcome

    batch = BatchOutcome(
        outcomes=[
            RecognizeOutcome(source="a.jpg", results=[_plate_result(PlateColor.YELLOW, "京A1")]),
            RecognizeOutcome(source="b.jpg", results=[_plate_result(PlateColor.HAZMAT, "京A2")]),
            RecognizeOutcome(source="c.jpg", results=[_plate_result(PlateColor.BLUE, "京A3")]),
        ],
        started=0.0,
    )
    batch.keep_colors({PlateColor.YELLOW, PlateColor.HAZMAT})
    assert sorted(batch.all_texts) == ["京A1", "京A2"]


def test_batch_keep_colors_is_chainable() -> None:
    from lpr.api import BatchOutcome

    batch = BatchOutcome(outcomes=[], started=0.0)
    assert batch.keep_colors({PlateColor.BLUE}) is batch


def test_batch_color_summary_counts_each_color() -> None:
    from lpr.api import BatchOutcome, RecognizeOutcome

    batch = BatchOutcome(
        outcomes=[
            RecognizeOutcome(
                source="a.jpg",
                results=[
                    _plate_result(PlateColor.YELLOW),
                    _plate_result(PlateColor.HAZMAT),
                ],
            ),
            RecognizeOutcome(source="b.jpg", results=[_plate_result(PlateColor.YELLOW)]),
        ],
        started=0.0,
    )
    summary = batch.color_summary()
    assert summary[PlateColor.YELLOW.label] == 2
    assert summary[PlateColor.HAZMAT.label] == 1


def test_batch_color_summary_empty() -> None:
    from lpr.api import BatchOutcome

    assert BatchOutcome(outcomes=[], started=0.0).color_summary() == {}


# ══════════════════════════════════════════════ 包级导出


def test_public_exports() -> None:
    import lpr

    assert lpr.parse_color("危险品") is PlateColor.HAZMAT
    assert set(lpr.COLOR_NAMES) == {"黑色", "蓝色", "绿色", "白色", "黄色"}


def test_refine_by_color_supports_hazmat() -> None:
    """回归：HAZMAT 没有 HSV 区间，早期实现会在投影裁剪时 KeyError 崩溃。

    危险品牌应以「黑底」为基准做投影（字是前景，不参与裁剪）。
    """
    from lpr.engines.locate import _color_mask, refine_by_color

    cfg = LocateConfig()
    img = _plate((10, 10, 10), (0, 220, 255))

    # 不应抛 KeyError
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    assert cv2.countNonZero(_color_mask(hsv, PlateColor.HAZMAT)) > 0

    out = refine_by_color(img, PlateColor.HAZMAT, cfg)
    assert out is not None
    assert out.size > 0


def test_refine_by_color_all_colors_do_not_crash() -> None:
    """六种底色都应能安全走完投影裁剪。"""
    from lpr.engines.locate import refine_by_color

    cfg = LocateConfig()
    for color in PlateColor:
        out = refine_by_color(_plate((10, 10, 10), (255, 255, 255)), color, cfg)
        assert out is not None, f"{color} 返回了 None"


def test_hazmat_mask_uses_black_base() -> None:
    """危险品牌的掩码基准应是黑底，而非黄字。"""
    from lpr.engines.locate import _REFINE_BASE

    assert _REFINE_BASE[PlateColor.HAZMAT] is PlateColor.BLACK
