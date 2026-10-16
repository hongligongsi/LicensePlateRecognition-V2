"""补救预处理（去噪 / 对比度均衡 / 二值化）的测试。

设计约束（与实现里的注释一一对应）：

* **零回归**：只在主计划 6 轮一个合法读数都抽不出来时才触发 —— 已命中
  候选既不会多跑一次识别，也不会被改写；
* **有预算**：每张图最多补救 ``_RESCUE_BUDGET`` 个候选。识别是逐候选处理的，
  而候选池里常混着车身纹理之类的垃圾候选，它们本来就该读不出字，
  却会让补救路径一次次空跑（实测 deep 单图耗时 716ms → 1344ms 而命中不变）；
* **可独立降级**：某个变体处理失败只跳过它自己，不抛异常、不拖垮其余变体。
"""

from __future__ import annotations

import numpy as np
import pytest

import lpr.engines.deep_engine as deep_mod
from lpr.config import EngineConfig
from lpr.engines.deep_engine import (
    _RESCUE_BUDGET,
    _RESCUE_SCALE,
    _RESCUE_VARIANTS,
    DeepEngine,
    _rescue_preprocess,
)
from lpr.types import PlateColor

# ══════════════════════════════════════════════════ 纯函数


class TestRescuePreprocess:
    """三个变体的形状/通道契约与降级行为。"""

    def _roi(self, h: int = 60, w: int = 220) -> np.ndarray:
        # 造一张有内容的图：上白下黑会让 otsu 有阈值可分
        img = np.full((h, w, 3), 240, np.uint8)
        img[h // 2 :, :] = 20
        return img

    @pytest.mark.parametrize("variant", _RESCUE_VARIANTS)
    def test_returns_upscaled_bgr(self, variant: str) -> None:
        roi = self._roi()
        out = _rescue_preprocess(roi, variant)
        assert out is not None
        assert out.ndim == 3 and out.shape[2] == 3
        # 尺寸按 `_RESCUE_SCALE` 放大（int() 截断，允许 1px 误差）
        assert out.shape[0] == int(roi.shape[0] * _RESCUE_SCALE)
        assert out.shape[1] == int(roi.shape[1] * _RESCUE_SCALE)

    def test_unknown_variant_returns_none(self) -> None:
        assert _rescue_preprocess(self._roi(), "不存在的变体") is None

    @pytest.mark.parametrize("variant", _RESCUE_VARIANTS)
    def test_degenerate_input_returns_none(self, variant: str) -> None:
        """空图 / 1 像素图不能让 cv2 抛异常，只能返回 None 由调用方跳过。"""
        assert _rescue_preprocess(np.zeros((0, 0, 3), np.uint8), variant) is None
        assert _rescue_preprocess(np.zeros((1, 1, 3), np.uint8), variant) is None

    def test_memory_guard_returns_none(self) -> None:
        """放大后超过内存闸门的大图必须被拒（与 `_upscale` 同一道闸门）。"""
        big = np.zeros((2000, 2000, 3), np.uint8)
        assert _rescue_preprocess(big, "denoise") is None

    def test_otsu_output_is_binary(self) -> None:
        """二值化变体的产物只应有两档灰度值 —— 这是它"压掉底色"的意义所在。"""
        out = _rescue_preprocess(self._roi(), "otsu")
        assert out is not None
        unique = np.unique(out)
        assert set(unique.tolist()) <= {0, 255}
        # 灰度转 BGR：三个通道必须完全相同
        assert np.array_equal(out[:, :, 0], out[:, :, 1])
        assert np.array_equal(out[:, :, 1], out[:, :, 2])

    def test_clahe_keeps_channel_count(self) -> None:
        out = _rescue_preprocess(self._roi(), "clahe")
        assert out is not None and out.shape[2] == 3

    def test_budget_is_positive(self) -> None:
        """预算必须 ≥1，否则补救路径永远不触发（等于功能被关掉）。"""
        assert _RESCUE_BUDGET >= 1

    def test_variants_are_all_implemented(self) -> None:
        """`_RESCUE_VARIANTS` 里不能有 `_rescue_preprocess` 不认识的拼写。"""
        roi = self._roi()
        for variant in _RESCUE_VARIANTS:
            assert _rescue_preprocess(roi, variant) is not None, variant


# ══════════════════════════════════════════════════ 接线


@pytest.fixture
def engine_stub():
    """不加载任何 OCR 模型的引擎骨架（与 test_secondary_cross_validation 同法）。"""
    eng = DeepEngine.__new__(DeepEngine)
    eng.config = EngineConfig()
    eng.cfg = eng.config.locate
    eng.backend = "rapidocr"
    eng._models_dir = None
    eng._ocr = None
    eng._recognizer = None
    eng._recognizer2 = None
    eng._impl = "rapidocr"
    eng._locator_ready = False
    eng._rec_only_supported = True
    eng._ready = True
    return eng


class TestRescueWiring:
    """补救路径的触发条件与预算。"""

    def _roi(self) -> np.ndarray:
        return np.full((60, 220, 3), 255, np.uint8)

    def _call(self, eng, n_candidates: int):
        rois = [(self._roi(), PlateColor.BLUE, (0, 0, 220, 60)) for _ in range(n_candidates)]
        return eng._recognize_candidates(rois)

    def test_rescue_when_primary_silent(self, engine_stub, monkeypatch) -> None:
        """主计划 6 轮全空 → 补救变体读出的号码被采纳，并打上标记。"""
        eng = engine_stub
        calls: list[str] = []

        def fake_rec(img):
            # 主计划（放大+锐化后的图）永远读不出；补救（3 倍拉伸）能读出
            if img.shape[1] >= 600:
                return [("京A12345", 0.9, [0.9] * 7)]
            return []

        monkeypatch.setattr(eng, "_recognize_only_detailed", fake_rec)
        # 补救预处理是**模块级函数**，必须打在模块上才拦得住（打在实例上无效）
        monkeypatch.setattr(
            deep_mod,
            "_rescue_preprocess",
            lambda roi, variant: (calls.append(variant), self._roi_triple())[1],
        )

        results = self._call(eng, 1)

        assert [r.text for r in results] == ["京A12345"]
        assert results[0].meta["rescue_preprocess"] is True
        assert calls, "补救预处理必须被调用"

    def test_no_rescue_when_primary_has_votes(self, engine_stub, monkeypatch) -> None:
        """主计划有合法读数 → 补救路径一次都不许跑（零回归的保证）。"""
        eng = engine_stub
        monkeypatch.setattr(
            eng, "_recognize_only_detailed", lambda img: [("京A12345", 0.9, [0.9] * 7)]
        )
        calls: list[str] = []
        monkeypatch.setattr(
            deep_mod, "_rescue_preprocess", lambda roi, variant: calls.append(variant)
        )

        results = self._call(eng, 1)

        assert [r.text for r in results] == ["京A12345"]
        assert results[0].meta["rescue_preprocess"] is False
        assert not calls

    def test_budget_limits_rescue_attempts(self, engine_stub, monkeypatch) -> None:
        """候选再多，补救次数也不得超过 ``_RESCUE_BUDGET`` 个候选的份额。"""
        eng = engine_stub
        monkeypatch.setattr(eng, "_recognize_only_detailed", lambda img: [])
        calls: list[str] = []
        monkeypatch.setattr(
            deep_mod,
            "_rescue_preprocess",
            lambda roi, variant: calls.append(variant) or self._roi_triple(),
        )

        # 5 个候选全部读不出（垃圾候选的典型形态）
        assert self._call(eng, 5) == []
        assert len(calls) == _RESCUE_BUDGET * len(_RESCUE_VARIANTS)

    def test_rescue_marker_false_on_normal_path(self, engine_stub, monkeypatch) -> None:
        """正常命中的结果也要显式带上 ``rescue_preprocess=False``（便于审计）。"""
        eng = engine_stub
        monkeypatch.setattr(
            eng, "_recognize_only_detailed", lambda img: [("沪B99999", 0.9, [0.9] * 7)]
        )
        results = self._call(eng, 1)
        assert results[0].meta["rescue_preprocess"] is False

    def _roi_triple(self) -> np.ndarray:
        """补救变体返回的图（3 倍尺寸，宽度 ≥600 用来区分主计划）。"""
        return np.full((180, 660, 3), 255, np.uint8)
