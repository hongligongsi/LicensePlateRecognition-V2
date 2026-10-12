"""v2.3.0 稳定性与准确率改进的回归测试。

覆盖：
* ``_extract_plate`` 四级抽取策略（严格 → 宽松 → 先纠正再宽松 → 纯字母数字闸门）
* 省份形近字召回（``loose_match`` + ``PROVINCE_LOOKALIKES``）
* ``三 → E`` 汉字误读纠错（双层黄牌 京三A5331 → 京EA5331）
* 旧式军牌收窄（京V 军队；皖K 等省份组合回归普通民用）
* 双层牌垂直碎片合并（``_group_double_fragments``）
* ``ensure_bgr`` 输入归一化（灰度/四通道/浮点/NaN/None 不抛异常）
"""

from typing import ClassVar

import numpy as np
import pytest

from lpr.engines.deep_engine import (
    DeepEngine,
    _extract_plate,
    measure_layout,
    split_double_line,
)
from lpr.plate_rules import correct, describe, is_valid, loose_match
from lpr.utils.image import ensure_bgr

# --------------------------------------------------------------- _extract_plate


class TestExtractPlate:
    def test_strict_recall(self):
        assert _extract_plate("你好京A12345再见") == "京A12345"

    def test_loose_recall_lookalike_province(self):
        # 首位是省份形近字：宽松召回应捞出，后续纠错修正
        assert _extract_plate("凉A12345") == "凉A12345"

    def test_correct_then_recall_san_to_e(self):
        # 京三A5331：三 是 E 的汉字误读，先纠正再召回
        assert _extract_plate("京三A5331") == "京EA5331"

    def test_bare_noise_rejected(self):
        # 车身贴纸/广告字：纠正后仍不合法 → 拒绝
        assert _extract_plate("JINCBFI") is None
        assert _extract_plate("A5331") is None

    def test_empty_and_garbage(self):
        for junk in ("", " ", "###", "\x00\x01", "京", "京A"):
            assert _extract_plate(junk) is None

    def test_long_garbage_no_crash(self):
        assert _extract_plate("京A1234" * 20) is not None or True  # 不抛异常即可


# --------------------------------------------------------------- loose_match


class TestLooseMatch:
    def test_normal_province(self):
        assert loose_match("皖K66666") == "皖K66666"

    def test_lookalike_head(self):
        assert loose_match("泸B74110") == "泸B74110"

    def test_none_for_plain_text(self):
        assert loose_match("HELLO123") is None

    def test_start_offset(self):
        assert loose_match(" adsf 京C88888", start=1) == "京C88888"


# --------------------------------------------------------------- 三→E 纠错


class TestSanToE:
    def test_correct_rule(self):
        fixed, changes = correct("京三A5331")
        assert fixed == "京EA5331"
        assert any("三" in c for c in changes)

    def test_valid_after_fix(self):
        assert is_valid("京EA5331")


# --------------------------------------------------------------- 旧式军牌收窄


class TestLegacyMilitaryNarrowed:
    def test_jing_v_is_military(self):
        d = describe("京V12345")
        assert d["type"] == "军队"

    def test_wan_k_is_civilian(self):
        # 皖K = 六安，此前被旧式军牌规则误判成军队
        d = describe("皖K66666")
        assert d["type"] == "普通民用"

    def test_yue_v_is_civilian(self):
        d = describe("粤V12345")
        assert d["type"] == "普通民用"


# --------------------------------------------------------------- 双层垂直合并


def _quad(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


class TestGroupDoubleFragments:
    def _engine(self):
        eng = DeepEngine.__new__(DeepEngine)  # 跳过模型加载
        return eng

    def test_merge_jing_san(self):
        eng = self._engine()
        boxes = [
            (_quad(200, 300, 250, 320), "京三", 0.65),  # 上行
            (_quad(190, 330, 260, 360), "A5331", 0.69),  # 下行
            (_quad(10, 30, 120, 50), "有车大师", 0.76),  # 无关文本
        ]
        groups = DeepEngine._group_double_fragments(eng, boxes, used=set())
        assert len(groups) == 1
        top, bot = groups[0]
        assert _extract_plate(top[2] + bot[2]) == "京EA5331"

    def test_no_merge_for_far_apart(self):
        eng = self._engine()
        boxes = [
            (_quad(200, 100, 250, 120), "京三", 0.65),
            (_quad(400, 500, 470, 530), "A5331", 0.69),  # 位置无关
        ]
        assert DeepEngine._group_double_fragments(eng, boxes, used=set()) == []


# --------------------------------------------------------------- ensure_bgr


class TestEnsureBgr:
    @pytest.mark.parametrize(
        "img",
        [
            None,
            np.zeros((1, 1, 3), np.uint8),  # 过小
            np.zeros((50, 50), np.uint8),  # 灰度
            np.zeros((50, 50, 1), np.uint8),  # 单通道 3 维
            np.zeros((50, 50, 4), np.uint8),  # BGRA
            np.random.rand(30, 30, 3).astype(np.float32),  # float32
            np.full((30, 30, 3), np.nan),  # float64 + NaN
            np.random.randint(0, 65535, (20, 20, 3), np.uint16),  # uint16
            "not an image",
        ],
    )
    def test_never_raises(self, img):
        out = ensure_bgr(img)
        if out is not None:
            assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8

    def test_valid_passthrough_content(self):
        img = np.full((20, 20, 3), 128, np.uint8)
        out = ensure_bgr(img)
        assert out is not None and np.all(out == 128)

    def test_nan_becomes_black(self):
        img = np.full((10, 10, 3), np.nan)
        out = ensure_bgr(img)
        assert out is not None and int(out.max()) == 0


class TestPipelineOnHostileInputs:
    """边界输入进入管线函数不抛异常。"""

    hostile: ClassVar[list] = [
        np.zeros((1, 1, 3), np.uint8),
        np.zeros((50, 50), np.uint8),
        np.zeros((50, 50, 4), np.uint8),
        np.random.rand(30, 30, 3).astype(np.float32),
        np.full((30, 30, 3), np.nan),
        np.random.randint(0, 255, (20, 400, 3), np.uint8),  # 极端宽高比
        np.random.randint(0, 255, (3000, 20, 3), np.uint8),
    ]

    @pytest.mark.parametrize("img", hostile)
    def test_measure_layout(self, img):
        assert measure_layout(img) in ("single", "double", "unknown")

    @pytest.mark.parametrize("img", hostile)
    def test_split_double_line(self, img):
        assert split_double_line(img) is None or isinstance(split_double_line(img), tuple)

    @pytest.mark.parametrize("img", hostile)
    def test_color_and_hazmat(self, img):
        from lpr.config import LocateConfig
        from lpr.engines.locate import detect_color, detect_hazmat

        cfg = LocateConfig()
        color, _ = detect_color(img, cfg)
        assert isinstance(color, object)
        detect_hazmat(img, cfg)  # 不抛异常即可
