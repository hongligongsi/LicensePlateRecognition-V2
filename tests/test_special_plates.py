"""军牌/警牌/白牌等特种号牌的支持回归。

背景（v2.3.2 改进）：特种牌曾是全链路最弱的一环——

* 严格抽取正则缺「使/领」末位，使领馆牌被截断成非法残串；
* 武警牌在严格层完全没有分支（只能靠宽松召回兜底）；
* ``correct()`` 对军队/使领馆/武警字头整体早退，序号位 I/O 从不纠正；
* 特殊后缀（警/挂/学/危/使/领/港/澳）与字头（军/空/海…）没有形近字表，
  OCR 认错一个字整条号码直接判无效；
* ``correct()`` 会把**已经合法**的使领馆牌（京123使）纠成非法（京J23使）；
* 阴影下的警牌（img/police.jpg）白底被判黑牌——红色字符信号是唯一
  可靠的物理特征（军牌字头/警牌「警」/武警「WJ」均为红色）；
* cv 引擎的混淆修正会把 ``使123456`` 的数字位破坏成字母。
"""

from __future__ import annotations

import numpy as np
import pytest

from lpr.plate_rules import (
    HEAD_LOOKALIKES,
    SUFFIX_LOOKALIKES,
    correct,
    describe,
    identify_format,
    is_valid,
    plate_score,
)
from lpr.types import PlateColor


# --------------------------------------------------------------------------- #
# 严格校验层：特种格式必须合法
# --------------------------------------------------------------------------- #


class TestSpecialFormats:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("京A1234警", "警用"),
            ("京A12345警", "警用"),
            ("军A12345", "军队"),
            ("空B12345", "军队"),
            ("海C12345", "军队"),
            ("京V12345", "军队"),  # 旧式军牌
            ("京123使", "使领馆"),
            ("京123领", "使领馆"),
            ("使123456", "使领馆"),
            ("WJ京1234X", "武警"),
            ("WJ京1234", "武警"),  # 无尾位字母的批次
            ("WJ0112345", "武警"),  # 旧式（WJ+2位机构码+5位序号）
            ("WJ012345", "武警"),  # 旧式（4位序号）
            ("粤Z1234港", "港澳入出境"),
            ("京A1234危", "危险品运输"),
            ("京A1234挂", "挂车"),
            ("京A123学", "教练车"),
        ],
    )
    def test_valid_special_formats(self, text: str, expected: str):
        fmt = identify_format(text)
        assert fmt is not None, f"{text} 应为合法的{expected}号牌"
        assert fmt.name == expected

    @pytest.mark.parametrize(
        "text",
        [
            "WJ1234X",     # 太短：WJ 后既非省份也非足够位数
            "军12345",     # 军牌第 2 位必须是字母
            "京A12345使",  # 使领馆不接受字母第 2 位（现行口径）
        ],
    )
    def test_still_invalid(self, text: str):
        assert not is_valid(text)


# --------------------------------------------------------------------------- #
# 严格抽取层（deep 引擎 _PLATE_TEXT）：特种牌必须能被完整抽出
# --------------------------------------------------------------------------- #


class TestStrictExtraction:
    """从 deep_engine 导入正则做黑盒验证（避免依赖引擎加载）。"""

    @pytest.fixture(autouse=True)
    def _regex(self):
        from lpr.engines.deep_engine import _PLATE_TEXT

        self.regex = _PLATE_TEXT

    def _extract(self, text: str) -> str | None:
        m = self.regex.search(text)
        return m.group(0) if m else None

    def test_embassy_with_ling_not_truncated(self):
        # 修复前：「京123领」被截成「京123」（残串既不合法也丢了后缀）
        assert self._extract("京123领") == "京123领"
        assert self._extract("京123使") == "京123使"

    def test_embassy_digit_only_strict(self):
        # 修复前：「使123456」在严格层抽不到
        assert self._extract("使123456") == "使123456"

    def test_armed_police_strict(self):
        # 修复前：武警牌严格层完全没有分支
        assert self._extract("WJ京1234X") == "WJ京1234X"
        assert self._extract("WJ京1234") == "WJ京1234"
        assert self._extract("WJ0112345") == "WJ0112345"

    def test_police_still_extracted(self):
        assert self._extract("京A1234警") == "京A1234警"

    def test_province_branch_unchanged(self):
        assert self._extract("京A12345") == "京A12345"
        assert self._extract("京AD77972") == "京AD77972"
        assert self._extract("民航A0001") == "民航A0001"


# --------------------------------------------------------------------------- #
# 纠错层：字头/后缀形近字 与 I/O 修正
# --------------------------------------------------------------------------- #


class TestCorrectionSpecial:
    def test_io_fixed_inside_military(self):
        # 修复前：correct() 对军队字头整体早退，O 从不纠正
        fixed, changes = correct("军A1O345")
        assert fixed == "军A10345"
        assert any("O → 0" in c for c in changes)

    def test_head_lookalike_military(self):
        fixed, changes = correct("罕A12345")
        assert fixed == "军A12345"
        assert any("罕" in c for c in changes)

    def test_head_lookalike_hai(self):
        fixed, _ = correct("每B12345")
        assert fixed == "海B12345"

    def test_head_lookalike_embassy(self):
        fixed, _ = correct("便123456")
        assert fixed == "使123456"

    def test_suffix_jing(self):
        fixed, _ = correct("京A1234敬")
        assert fixed == "京A1234警"

    def test_suffix_gua(self):
        # 「桂」是省份简称，但出现在末位只可能是「挂」
        fixed, _ = correct("京A1234桂")
        assert fixed == "京A1234挂"

    def test_suffix_xue(self):
        fixed, _ = correct("京A123宇")
        assert fixed == "京A123学"

    def test_suffix_wei(self):
        fixed, _ = correct("京A1234免")
        assert fixed == "京A1234危"

    def test_suffix_ling(self):
        fixed, _ = correct("京123顿")
        assert fixed == "京123领"

    def test_valid_input_untouched(self):
        # 关键回归：已合法的号码一个字符都不能动。
        # 修复前「京123使」会被"第 2 位必须字母"规则纠成非法的 京J23使。
        for text in ("京123使", "京123领", "京A1234警", "京A12345", "WJ0112345"):
            fixed, changes = correct(text)
            assert fixed == text, f"{text} 已合法，不应改动（得到 {fixed}）"
            assert changes == []

    def test_lookalike_gate_requires_valid_plate(self):
        # 形近字替换必须整体合法才采纳：孤立改字不允许
        # 「敬A12345」首位敬无对应省份/字头合法归宿，不应被乱改
        fixed, _ = correct("敬A12345")
        assert fixed == "敬A12345"

    def test_lookalike_tables_disjoint_positions_ok(self):
        # 「字/宇」末位→学、首位→军：按位置取表，两张表允许同字不同义
        assert SUFFIX_LOOKALIKES["字"] == "学"
        assert HEAD_LOOKALIKES["字"] == "军"


# --------------------------------------------------------------------------- #
# 打分层：特种牌不再被系统性压低
# --------------------------------------------------------------------------- #


class TestScoring:
    def test_special_plates_score_full(self):
        # 合法特种牌应拿到与普通牌同级的分数
        for text in ("京A1234警", "军A12345", "WJ京1234X", "京123使"):
            assert plate_score(text) >= 0.8

    def test_describe_type(self):
        assert describe("京A1234警")["type"] == "警用"
        assert describe("军A12345")["type"] == "军队"
        assert describe("WJ京1234X")["type"] == "武警"


# --------------------------------------------------------------------------- #
# 定位判色：红色字符信号（军/警/武警白牌）
# --------------------------------------------------------------------------- #


def _make_white_plate_with_red_char(
    red_ratio_target: float = 0.06,
    char_height_frac: float = 0.8,
    bg_value: int = 120,  # 阴影下的白底（灰度 ~120，进不了 V>=200 的白区间）
) -> np.ndarray:
    """合成一块"阴影白底 + 黑字 + 红色字符"的警牌式 ROI。"""
    h, w = 60, 220
    img = np.full((h, w, 3), (bg_value, bg_value, bg_value), np.uint8)
    # 黑色字符 5 个
    for i in range(5):
        x0 = 10 + i * 30
        img[10 : h - 10, x0 : x0 + 18] = 20
    # 红色字符（末位，与字符等高）
    red_h = int(h * char_height_frac)
    red_w = 18
    area = red_h * red_w
    need = int(red_ratio_target * h * w)
    red_w_actual = max(red_w, need // max(1, red_h))
    img[(h - red_h) // 2 : (h + red_h) // 2, w - red_w_actual - 5 : w - 5] = (
        40, 40, 220,  # BGR 红色
    )
    return img


class TestRedSignalColor:
    @pytest.fixture
    def cfg(self):
        from lpr.config import LocateConfig

        return LocateConfig()

    def test_shaded_police_plate_is_white(self, cfg):
        from lpr.engines.locate import detect_color

        roi = _make_white_plate_with_red_char()
        color, ratio = detect_color(roi, cfg)
        assert color is PlateColor.WHITE
        assert ratio >= 0.5

    def test_no_red_stays_black(self, cfg):
        from lpr.engines.locate import detect_color

        roi = _make_white_plate_with_red_char()
        # 把红色字符涂黑 → 无红信号，行为与旧版一致（此处黑白占比黑多判黑）
        roi[:, -25:-5] = 20
        color, _ = detect_color(roi, cfg)
        assert color is not PlateColor.WHITE

    def test_scarred_red_noise_ignored(self, cfg):
        from lpr.engines.locate import detect_color

        # 红色块高度不足 45%（散乱反射特征）→ 不触发白牌判定
        roi = _make_white_plate_with_red_char(char_height_frac=0.3)
        color, _ = detect_color(roi, cfg)
        assert color is not PlateColor.WHITE

    def test_dominant_red_ignored(self, cfg):
        from lpr.engines.locate import detect_color

        # 红色占比超过上限（大片红色车身/招牌）→ 不触发
        roi = _make_white_plate_with_red_char(red_ratio_target=0.6)
        color, _ = detect_color(roi, cfg)
        assert color is not PlateColor.WHITE


# --------------------------------------------------------------------------- #
# cv 引擎：混淆修正不再破坏特种字头
# --------------------------------------------------------------------------- #


class TestCvEngineFixConfusions:
    @pytest.fixture(autouse=True)
    def _fix(self):
        from lpr.engines.cv_engine import _fix_confusions

        self.fix = _fix_confusions

    def test_embassy_digit_second_char_untouched(self):
        # 修复前：使123456 的第 2 位数字被强行字母化成 使J23456
        assert self.fix("使123456") == "使123456"

    def test_armed_police_untouched(self):
        assert self.fix("WJ京1234X") == "WJ京1234X"

    def test_military_second_char_kept(self):
        assert self.fix("军A12345") == "军A12345"

    def test_province_digit_second_char_letterized(self):
        # 省份牌第 2 位仍须字母
        assert self.fix("京1A12345")[1].isalpha()
        assert self.fix("京0A12345")[1].isalpha()


class TestCvEngineScore:
    def test_special_plate_not_penalized(self):
        from lpr.engines.cv_engine import _score_text

        normal = _score_text("京A12345")
        police = _score_text("京A1234警")
        military = _score_text("军A12345")
        assert police == normal
        assert military == normal

    def test_garbage_still_low(self):
        from lpr.engines.cv_engine import _score_text

        assert _score_text("A1B2C3") < _score_text("京A12345")
