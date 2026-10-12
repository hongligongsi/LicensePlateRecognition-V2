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
            "WJ1234X",  # 太短：WJ 后既非省份也非足够位数
            "军12345",  # 军牌第 2 位必须是字母
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
    red_h * red_w
    need = int(red_ratio_target * h * w)
    red_w_actual = max(red_w, need // max(1, red_h))
    img[(h - red_h) // 2 : (h + red_h) // 2, w - red_w_actual - 5 : w - 5] = (
        40,
        40,
        220,  # BGR 红色
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


# --------------------------------------------------------------------------- #
# 号牌规范口径核对（v2.3.4）：三处曾被怀疑"我们写错了"的地方
# --------------------------------------------------------------------------- #


class TestNormsEmbassy:
    """使领馆号牌：**「使」字在号码末尾**（2017-01-09 起）。

    依据《外交车辆管理办法》及 GA 36-2018 对使馆/领馆汽车号牌式样的修订：
    2017 年换发时把原先居首的**红色**「使」字改为**白色**并移到号码**之后**。

    参考实现用 ``[0-9]{6}使`` 是对的，我们原先只有 ``使[0-9]{6}``（旧式），
    那条被当成"也许我们才对"——核对后确认**对方是对的**。
    """

    @pytest.mark.parametrize("text", ["123456使", "X59301使"])
    def test_current_style_digits_then_shi(self, text: str):
        assert identify_format(text) is not None
        assert identify_format(text).name == "使领馆"

    def test_legacy_style_still_accepted(self):
        # 2017-05-01 废止，但旧影像里仍有
        assert identify_format("使123456").name == "使领馆"

    @pytest.mark.parametrize("text", ["粤204领", "沪204领"])
    def test_consulate_province_code_ling(self, text: str):
        # 领馆牌 = 省级简称 + 国家/国际组织代码 + 「领」（没有纯数字形态）
        assert identify_format(text).name == "使领馆"

    def test_consulate_has_no_bare_digit_form(self):
        # 「领」必须带省份简称：纯数字+领 不是领馆牌的形态
        assert identify_format("123456领") is None

    def test_current_style_not_truncated_by_extraction(self):
        from lpr.engines.deep_engine import _PLATE_TEXT

        assert _PLATE_TEXT.search("123456使").group(0) == "123456使"


class TestNormsPolice:
    """警用号牌：**第三位可以是 A/B/C/D**，不只是数字。

    依据《警车管理规定》附件 2《警车号牌式样》：登记编号七位——第一位省简称、
    第二位地区字母、**第三位 A/B/C/D 或阿拉伯数字**（依次为法院、检察院、
    国家安全机关、司法行政机关、公安机关），第四至六位注册编号，第七位红色"警"。
    """

    @pytest.mark.parametrize(
        "text,department",
        [
            ("京AA006警", "法院"),
            ("京AB006警", "检察院"),
            ("京AC006警", "国家安全机关"),
            ("京AD006警", "司法行政机关"),
        ],
    )
    def test_letter_third_position_is_legal(self, text: str, department: str):
        fmt = identify_format(text)
        assert fmt is not None, f"第三位字母的警牌（{department}）应合法"
        assert fmt.name == "警用"

    @pytest.mark.parametrize("text", ["京A0006警", "京A1234警", "京A12345警"])
    def test_digit_third_position_still_legal(self, text: str):
        assert identify_format(text).name == "警用"

    def test_only_abcd_admitted(self):
        # 第三位的字母只到 D：E 之后不属警用部门代码
        assert identify_format("京AE006警") is None

    def test_extraction_keeps_letter_third_position(self):
        from lpr.engines.deep_engine import _PLATE_TEXT

        assert _PLATE_TEXT.search("京AA006警").group(0) == "京AA006警"


class TestNormsMilitary:
    """军队号牌：现行 **2012 式** 是「双字母 + 5 位数字」，汉字字头是 2004 式旧牌。

    2013-05-01 全军统一换发 2012 式：字头取军队大单位简称的**大写汉语拼音
    字母**、字头号为二级单位的大写英文字母、序号 5 位数字。
    我们原先只有「汉字字头 + 字母 + 5 位数字」（2004 式，已废止），
    等于把现行军牌整体判成非法。
    """

    @pytest.mark.parametrize("text", ["LB12345", "KA12345", "HA12345", "VA12345"])
    def test_2012_style_two_letters(self, text: str):
        fmt = identify_format(text)
        assert fmt is not None, f"{text} 是现行 2012 式军牌"
        assert fmt.name == "军队"

    @pytest.mark.parametrize("text", ["空B12345", "军A12345"])
    def test_2004_style_kept_for_old_footage(self, text: str):
        assert identify_format(text).name == "军队"

    @pytest.mark.parametrize("text", ["AB1234", "AB123456", "A12345"])
    def test_letter_digit_shape_is_strict(self, text: str):
        # 必须是"恰好 2 字母 + 5 数字"：位数不对就不是军牌
        assert identify_format(text) is None

    def test_does_not_swallow_armed_police(self):
        from lpr.engines.deep_engine import _PLATE_TEXT

        # 纯字母数字分支若放在武警分支之前，会把 WJ012345 切成 WJ01234
        assert _PLATE_TEXT.search("WJ012345").group(0) == "WJ012345"
        assert identify_format("WJ012345").name == "武警"


class TestNormsHazmat:
    """危险品运输号牌：末位「危」，并容忍 OCR 多认成「危险/危险品」。"""

    @pytest.mark.parametrize("text", ["京A1234危", "京A1234危险", "京A1234危险品"])
    def test_suffix_variants_all_legal(self, text: str):
        fmt = identify_format(text)
        assert fmt is not None
        assert fmt.name == "危险品运输"

    def test_suffix_variants_extractable(self):
        from lpr.engines.deep_engine import _PLATE_TEXT

        for text in ("京A1234危", "京A1234危险", "京A1234危险品"):
            assert _PLATE_TEXT.search(text).group(0) == text

    def test_bare_xian_is_not_a_suffix(self):
        # 「险」不能单独当后缀（放宽的只是"危"后面的尾巴）
        assert identify_format("京A1234险") is None

    def test_plain_plate_unaffected(self):
        assert identify_format("京A12345").name == "普通民用"
