# -*- coding: utf-8 -*-
"""12 类车牌支持 + 双层版式维度的测试。

覆盖用户点名的 12 类：

1. 单行蓝牌  2. 单行黄牌  3. 新能源车牌  4. 白色警用车牌  5. 教练车牌
6. 武警车牌  7. 双层黄牌  8. 双层白牌    9. 使领馆车牌    10. 港澳粤Z牌
11. 双层绿牌 12. 民航车牌

其中 7/8/11 与 1/2/3 的**号码**相同（双层是版式维度，不是号码维度），
因此这些用例同时验证版式字段的表达。
"""
from __future__ import annotations

import numpy as np
import pytest

from lpr.plate_rules import (
    CIVIL_AVIATION_HEADS,
    DEFAULT_LAYOUT,
    describe,
    identify_format,
    is_valid,
    plate_score,
)
from lpr.types import (
    COLOR_SWATCH,
    PlateColor,
    PlateLayout,
    PlateResult,
    parse_layout,
)

# ---------------------------------------------------------------- 用户 12 类逐项


class TestTwelvePlateKinds:
    """用户点名的 12 类车牌，逐项验证类型判定。"""

    CASES = [
        ("单行蓝牌", "京A12345", "普通民用"),
        ("单行黄牌", "京A12345", "普通民用"),
        ("新能源车牌", "京AD77972", "新能源"),
        ("白色警用车牌", "京A1234警", "警用"),
        ("教练车牌", "京A123学", "教练车"),
        ("武警车牌", "WJ京1234X", "武警"),
        ("双层黄牌", "京A12345", "普通民用"),
        ("双层白牌", "京A1234警", "警用"),
        ("使领馆车牌", "使123456", "使领馆"),
        ("港澳粤Z牌", "粤Z1234港", "港澳入出境"),
        ("双层绿牌", "京AD77972", "新能源"),
        ("民航车牌", "民航A0001", "民航"),
    ]

    @pytest.mark.parametrize(("label", "text", "expected"), CASES, ids=[c[0] for c in CASES])
    def test_kind_identified(self, label, text, expected):
        fmt = identify_format(text)
        assert fmt is not None, f"{label}({text}) 应被识别，实际为未知"
        assert fmt.name == expected

    def test_all_twelve_valid(self):
        for _label, text, _expected in self.CASES:
            assert is_valid(text)


# ---------------------------------------------------------------- 民航车牌细节


class TestCivilAviation:
    def test_all_management_bureaus(self):
        for letter in CIVIL_AVIATION_HEADS:
            assert identify_format(f"民航{letter}0001") is not None

    def test_h_is_not_a_bureau(self):
        # H 不在管理局代号内（A-G 共 7 个管理局）
        assert identify_format("民航H0001") is None

    def test_digit_count(self):
        assert is_valid("民航A1234")
        assert not is_valid("民航A123")     # 3 位数字不够
        assert not is_valid("民航A12345")   # 5 位数字太多

    def test_letters_not_allowed_in_serial(self):
        assert not is_valid("民航A12B4")

    def test_describe_flags(self):
        d = describe("民航B2345")
        assert d["type"] == "民航"
        assert d["is_civil_aviation"] is True
        assert d["valid"] is True

    def test_describe_non_aviation_flag_false(self):
        assert describe("京A12345")["is_civil_aviation"] is False

    def test_not_mistaken_for_province_plate(self):
        # "民" 不是省份简称，普通民用规则不能吞掉民航牌
        assert identify_format("民航A0001").name == "民航"

    def test_score_comparable_to_other_plates(self):
        assert plate_score("民航A0001") > 0.6


# ---------------------------------------------------------------- 版式维度


class TestPlateLayout:
    def test_enum_values(self):
        assert PlateLayout.SINGLE.value == "single"
        assert PlateLayout.DOUBLE.value == "double"
        assert PlateLayout.UNKNOWN.value == "unknown"

    def test_labels(self):
        assert PlateLayout.SINGLE.label == "单行"
        assert PlateLayout.DOUBLE.label == "双层"

    def test_parse_layout_chinese(self):
        assert parse_layout("单行") is PlateLayout.SINGLE
        assert parse_layout("双层") is PlateLayout.DOUBLE
        assert parse_layout("双行") is PlateLayout.DOUBLE

    def test_parse_layout_english(self):
        assert parse_layout("single") is PlateLayout.SINGLE
        assert parse_layout("double") is PlateLayout.DOUBLE

    def test_parse_layout_rejects_bad_input(self):
        with pytest.raises(ValueError):
            parse_layout("三层")

    def test_result_carries_layout(self):
        r = PlateResult(text="京A12345", layout=PlateLayout.DOUBLE)
        d = r.to_dict()
        assert d["layout"] == "double"
        assert d["layout_label"] == "双层"

    def test_result_default_layout_unknown(self):
        r = PlateResult(text="京A12345")
        assert r.layout is PlateLayout.UNKNOWN

    def test_str_shows_layout_when_known(self):
        r = PlateResult(text="京A12345", layout=PlateLayout.DOUBLE, confidence=0.9)
        assert "双层" in str(r)

    def test_str_omits_unknown_layout(self):
        r = PlateResult(text="京A12345", confidence=0.9)
        # 未知版式不额外标注（颜色标签里的"未知"是另一个维度，不受影响）
        assert "单行" not in str(r) and "双层" not in str(r)


class TestDescribeLayoutHint:
    def test_hazmat_double(self):
        assert describe("京A1234危")["layout_hint"] == "double"

    def test_trailer_double(self):
        assert describe("京A1234挂")["layout_hint"] == "double"

    def test_common_single(self):
        assert describe("京A12345")["layout_hint"] == "single"

    def test_unknown_format_no_hint(self):
        assert describe("XX999")["layout_hint"] is None

    def test_every_known_format_has_hint(self):
        from lpr.plate_rules import FORMATS

        for fmt in FORMATS:
            assert fmt.name in DEFAULT_LAYOUT, f"格式 {fmt.name} 缺少版式先验"


# ---------------------------------------------------------------- 版式图像判定


def _synth_plate(w: int, h: int, rows: int) -> np.ndarray:
    """合成车牌图：rows=1 单行 7 字符；rows=2 上 4 下 6 双行。"""
    img = np.full((h, w, 3), 200, np.uint8)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (0, 0, 0), 2)
    if rows == 1:
        for i in range(7):
            x = int(w * 0.08 + i * w * 0.125)
            cv2.rectangle(
                img, (x, int(h * 0.25)), (x + int(w * 0.09), int(h * 0.78)), (0, 0, 0), -1
            )
    else:
        for i in range(4):
            x = int(w * 0.2 + i * w * 0.17)
            cv2.rectangle(
                img, (x, int(h * 0.08)), (x + int(w * 0.12), int(h * 0.42)), (0, 0, 0), -1
            )
        for i in range(6):
            x = int(w * 0.08 + i * w * 0.15)
            cv2.rectangle(
                img, (x, int(h * 0.58)), (x + int(w * 0.1), int(h * 0.92)), (0, 0, 0), -1
            )
    return img


import cv2  # noqa: E402  放在模块级常量之后以便阅读

from lpr.engines.deep_engine import measure_layout, split_double_line  # noqa: E402


class TestLayoutMeasurement:
    def test_single_line_synthetic(self):
        # 单行牌真实比例 440x140（约 3.14:1）
        assert measure_layout(_synth_plate(440, 140, 1)) == "single"

    def test_double_line_synthetic(self):
        # 双行牌真实比例 440x220（约 2:1）
        assert measure_layout(_synth_plate(440, 220, 2)) == "double"

    def test_framed_single_not_mistaken_for_double(self):
        # 带上下边框的单行牌曾被误拆成双行 —— 回归用例
        assert measure_layout(_synth_plate(440, 140, 1)) == "single"
        assert split_double_line(_synth_plate(440, 140, 1)) is None

    def test_too_wide_is_never_double(self):
        # 宽高比远超双行牌上限，即便投影切出两段也不算双层
        img = _synth_plate(880, 140, 2)  # 6.3:1
        assert measure_layout(img) != "double"

    def test_empty_roi(self):
        assert measure_layout(np.empty((0, 0, 3), np.uint8)) == "unknown"

    def test_tiny_roi(self):
        assert measure_layout(np.zeros((4, 4, 3), np.uint8)) == "unknown"

    def test_gray_input(self):
        gray = cv2.cvtColor(_synth_plate(440, 220, 2), cv2.COLOR_BGR2GRAY)
        assert measure_layout(gray) == "double"

    def test_split_returns_two_bands(self):
        pair = split_double_line(_synth_plate(440, 220, 2))
        assert pair is not None
        top, bottom = pair
        assert top.shape[0] > 0 and bottom.shape[0] > 0

    def test_split_single_returns_none(self):
        assert split_double_line(_synth_plate(440, 140, 1)) is None


# ---------------------------------------------------------------- 底色回归


class TestColorRegression:
    def test_six_colors_have_swatch(self):
        for color in (
            PlateColor.BLACK,
            PlateColor.BLUE,
            PlateColor.GREEN,
            PlateColor.WHITE,
            PlateColor.YELLOW,
            PlateColor.HAZMAT,
        ):
            assert color in COLOR_SWATCH

    def test_green_covers_aviation(self):
        # 民航牌为绿底白字，落在 GREEN 底色下
        assert PlateColor.GREEN.label == "绿牌（新能源）"
