"""低置信交叉验证（第二识别模型）的测试。

交叉验证的设计约束：
* 默认关闭，未配置/模型缺失时主引擎行为完全不变；
* 只在主模型对某候选**完全无产出**时激活，已命中路径零回归风险；
* 第二模型读数按 ``secondary_vote_weight`` 折扣计票。
"""

from __future__ import annotations

import numpy as np
import pytest

from lpr.config import EngineConfig
from lpr.engines.deep_engine import DeepEngine
from lpr.types import PlateColor

pytestmark = pytest.mark.slow


@pytest.fixture
def engine_cls_patched(monkeypatch):
    """绕过真实 RapidOCR 加载，得到"主模型失败 + 第二模型成功"的引擎。"""
    eng = DeepEngine.__new__(DeepEngine)
    eng.config = EngineConfig(secondary_rec_model="models/whatever.onnx")
    eng.cfg = eng.config.locate
    eng.backend = "rapidocr"
    eng._models_dir = None
    eng._ocr = None
    eng._recognizer = None
    eng._impl = "rapidocr"
    eng._locator_ready = False
    eng._rec_only_supported = True
    eng._ready = True
    return eng


class TestSecondaryLoading:
    """第二模型的加载与降级行为。"""

    def test_default_disabled(self):
        eng = DeepEngine()
        assert eng._recognizer2 is None

    def test_missing_model_file_keeps_primary_alive(self):
        eng = DeepEngine(EngineConfig(secondary_rec_model="models/no_such.onnx"))
        eng.load()  # 不抛异常
        assert eng._recognizer2 is None

    def test_real_model_loads_when_present(self):

        from lpr.config import PROJECT_ROOT

        model = PROJECT_ROOT / "models" / "ch_PP-OCRv4_rec_mobile.onnx"
        if not model.exists():
            pytest.skip("第二模型未下载（models/*.onnx 不入库，按需下载）")
        pytest.importorskip("rapidocr_onnxruntime")
        eng = DeepEngine(EngineConfig(secondary_rec_model="models/ch_PP-OCRv4_rec_mobile.onnx"))
        eng.load()
        assert eng._recognizer2 is not None


class TestSecondaryRescue:
    """主模型失败时的补救路径。"""

    def _roi(self) -> np.ndarray:
        return np.full((60, 220, 3), 255, np.uint8)

    def test_rescue_when_primary_silent(self, engine_cls_patched, monkeypatch):
        eng = engine_cls_patched
        # 主模型对所有变体都读不出
        monkeypatch.setattr(eng, "_recognize_only_detailed", lambda img: [])
        # 第二模型读出一个合法车牌
        eng._recognizer2 = object()  # 仅需非 None 哨兵
        monkeypatch.setattr(eng, "_recognize2_detailed", lambda img: [("京A12345", 0.9, [0.9] * 7)])

        results = eng._recognize_candidates([(self._roi(), PlateColor.BLUE, (0, 0, 220, 60))])

        assert [r.text for r in results] == ["京A12345"]
        assert results[0].meta["secondary_rescue"] is True

    def test_no_rescue_when_primary_has_votes(self, engine_cls_patched, monkeypatch):
        eng = engine_cls_patched
        monkeypatch.setattr(
            eng, "_recognize_only_detailed", lambda img: [("京A12345", 0.9, [0.9] * 7)]
        )
        called = []

        def fake_rec2(img):
            called.append(img)
            return [("沪B99999", 0.9, [0.9] * 7)]

        eng._recognizer2 = object()
        monkeypatch.setattr(eng, "_recognize2_detailed", fake_rec2)

        results = eng._recognize_candidates([(self._roi(), PlateColor.BLUE, (0, 0, 220, 60))])

        assert [r.text for r in results] == ["京A12345"]
        assert results[0].meta["secondary_rescue"] is False
        assert not called  # 主模型有产出，第二模型绝不被调用

    def test_no_secondary_configured_no_crash(self, engine_cls_patched, monkeypatch):
        eng = engine_cls_patched
        monkeypatch.setattr(eng, "_recognize_only_detailed", lambda img: [])
        eng._recognizer2 = None

        results = eng._recognize_candidates([(self._roi(), PlateColor.BLUE, (0, 0, 220, 60))])
        assert results == []
