"""车牌语法规则引擎测试。"""

from __future__ import annotations

import pytest

from lpr.plate_rules import (
    PlateFormat,
    correct,
    describe,
    identify_format,
    is_valid,
    plate_score,
)
from lpr.utils.security import PrivacyPolicy, mask_plate


# --------------------------------------------------------------- 格式识别

@pytest.mark.parametrize(
    "text,expected",
    [
        ("京A12345", "普通民用"),
        ("京AD77972", "新能源"),
        ("沪AD12345", "新能源"),
        ("京A1234警", "警用"),
        ("京A1234挂", "挂车"),
        ("京A123学", "教练车"),
        ("军A12345", "军队"),
        ("京V12345", "军队"),
        ("使123456", "使领馆"),
        ("粤Z1234港", "港澳入出境"),
        ("粤Z1234澳", "港澳入出境"),
    ],
)
def test_identify_format(text: str, expected: str) -> None:
    fmt = identify_format(text)
    assert fmt is not None, f"{text} 应被识别"
    assert fmt.name == expected, f"{text} 应为 {expected}，实际 {fmt.name}"


@pytest.mark.parametrize(
    "text,province,issuer",
    [
        ("京A12345", "京", "A"),
        ("粤B12345", "粤", "B"),
        ("川A12345", "川", "A"),
        ("京AD77972", "京", "A"),
    ],
)
def test_province_and_issuer(text: str, province: str, issuer: str) -> None:
    info = describe(text)
    assert info["province"] == province
    assert info["issuer"] == issuer


def test_military_head_has_no_province() -> None:
    info = describe("军A12345")
    assert info["type"] == "军队"
    assert info["province"] is None


def test_identify_format_returns_none_for_garbage() -> None:
    for bad in ("", "ABC1234", "12345", "京A12"):
        assert identify_format(bad) is None


def test_plate_format_match_method() -> None:
    fmt = identify_format("京A12345")
    assert isinstance(fmt, PlateFormat)
    assert fmt.match("京A12345")
    assert not fmt.match("京AD77972")


# --------------------------------------------------------------- 合法性

@pytest.mark.parametrize(
    "text",
    ["京A12345", "京A1234警", "京AD77972", "使123456", "粤Z1234港",
     "京A123学", "京A1234挂", "京V12345", "军A12345"],
)
def test_is_valid_true(text: str) -> None:
    assert is_valid(text), f"{text} 应合法"


@pytest.mark.parametrize(
    "text",
    [
        "ABC1234",        # 无省份汉字
        "京I12345",       # I 不作发牌机关代号
        "京O12345",       # O 不作发牌机关代号
        "京A12",          # 长度不足
        "京A1234567890",  # 过长
        "",
        "12345",
    ],
)
def test_is_valid_false(text: str) -> None:
    assert not is_valid(text), f"{text} 应不合法"


def test_plate_score_in_range() -> None:
    for text in ("京A12345", "京AD77972", "京X00000", "ABC123", ""):
        s = plate_score(text)
        assert 0.0 <= s <= 1.0, f"{text} 得分越界: {s}"


def test_valid_scores_higher_than_invalid() -> None:
    # 合法 8 位新能源牌应高于缺少省份字头的字符串
    assert plate_score("京AD77972") > plate_score("ABC1234")
    # 7 位合法牌应高于缺少省份字头的字符串
    assert plate_score("京A12345") > plate_score("ABC1234")
    assert plate_score("") == 0.0
    # 非法（省份字头后无字母）应低于合法牌
    assert plate_score("京1234567") < plate_score("京A12345")


def test_same_shape_plates_are_indistinguishable() -> None:
    """京AQ7797Q 与 京AD77972 同为合法 8 位新能源牌，串形无法区分。

    这是真实世界的固有歧义，只能靠 OCR 置信度与多帧投票消解，
    因此这里断言两者得分相等（而不是错误的"谁更高"）。
    """
    assert plate_score("京AQ7797Q") == plate_score("京AD77972")


# --------------------------------------------------------------- 纠错

def test_correct_confuses_letter_i_to_digit() -> None:
    fixed, changes = correct("京AI5619")
    assert fixed == "京A15619"
    assert changes


def test_correct_confuses_letter_o_to_digit() -> None:
    fixed, changes = correct("京A1234O")
    assert fixed == "京A12340"
    assert changes


def test_correct_second_position_digit_to_letter() -> None:
    fixed, changes = correct("京012345")
    assert fixed == "京D12345", f"实际 {fixed}"
    assert changes


def test_correct_keeps_valid_text_untouched() -> None:
    fixed, changes = correct("京AD77972")
    assert fixed == "京AD77972"
    assert changes == []


def test_correct_never_changes_length() -> None:
    for raw in ("京AI5619", "京A1234O", "京012345", "粤Z1234O"):
        fixed, _ = correct(raw)
        assert len(fixed) == len(raw), f"{raw} 长度被改变为 {fixed}"


def test_correct_reports_reason_for_unknown_head() -> None:
    fixed, changes = correct("X12345")
    assert fixed == "X12345"
    assert changes, "应说明首位不合法"


# --------------------------------------------------------------- 脱敏

def test_mask_plate_hides_middle_but_keeps_ends() -> None:
    policy = PrivacyPolicy(mask_output=True)
    masked = mask_plate("京AD77972", policy)
    assert masked.startswith("京")
    assert masked.endswith("72")
    assert "AD779" not in masked, f"中间部分未脱敏: {masked}"


def test_mask_plate_is_noop_without_policy() -> None:
    assert mask_plate("京AD77972") == "京AD77972"


def test_mask_plate_short_input_unchanged() -> None:
    policy = PrivacyPolicy(mask_output=True)
    assert mask_plate("京A1", policy) == "京A1"
    assert mask_plate("", policy) == ""


def test_mask_plate_is_not_reversible_for_digits() -> None:
    policy = PrivacyPolicy(mask_output=True)
    masked = mask_plate("京AD77972", policy)
    assert masked != "京AD77972"
    assert masked.count("*") >= 3
