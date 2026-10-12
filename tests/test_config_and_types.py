"""单元测试：配置、类型、工具函数。运行: pytest tests/ -v"""
from __future__ import annotations

import numpy as np
import pytest

from lpr.config import Config
from lpr.types import PlateColor, PlateResult
from lpr.utils.image import draw_result, ensure_odd, resize_max_side


class TestConfig:
    def test_default_valid(self):
        cfg = Config()
        cfg.engine.validate()  # 不应抛异常

    def test_blur_even_rejected(self):
        cfg = Config()
        cfg.engine.locate.blur_kernel = 4
        with pytest.raises(ValueError):
            cfg.engine.validate()

    def test_unknown_engine_rejected(self):
        cfg = Config()
        cfg.engine.default = "magic"
        with pytest.raises(ValueError):
            cfg.engine.validate()

    def test_aspect_order(self):
        cfg = Config()
        cfg.engine.locate.min_aspect = 6
        with pytest.raises(ValueError):
            cfg.engine.validate()

    def test_roundtrip(self, tmp_path):
        cfg = Config()
        cfg.engine.locate.max_side = 1600
        path = tmp_path / "config.json"
        cfg.save(path)

        loaded = Config.load(path)
        assert loaded.engine.locate.max_side == 1600

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LPR_ENGINE", "deep")
        cfg = Config.load(tmp_path_none := "/nonexistent/config.json")
        assert cfg.engine.default == "deep"


class TestTypes:
    def test_result_str(self):
        res = PlateResult(text="京A12345", color=PlateColor.BLUE, confidence=0.9)
        assert "京A12345" in str(res)

    def test_result_empty(self):
        res = PlateResult()
        assert not res.ok

    def test_to_dict_serializable(self):
        import json

        res = PlateResult(
            text="京A12345", color=PlateColor.BLUE, bbox=(1, 2, 3, 4), confidence=0.8
        )
        payload = json.dumps(res.to_dict(), ensure_ascii=False)
        assert "京A12345" in payload


class TestImageUtils:
    def test_ensure_odd(self):
        assert ensure_odd(4) == 5
        assert ensure_odd(3) == 3
        assert ensure_odd(0, minimum=1) == 1

    def test_resize_max_side_noop(self):
        img = np.zeros((300, 400, 3), np.uint8)
        out, scale = resize_max_side(img, 1000)
        assert out is img
        assert scale == 1.0

    def test_resize_max_side_down(self):
        img = np.zeros((1000, 2000, 3), np.uint8)
        out, scale = resize_max_side(img, 500)
        assert max(out.shape[:2]) == 500
        assert abs(scale - 0.25) < 1e-6

    def test_draw_result(self):
        img = np.full((300, 400, 3), 200, np.uint8)
        res = PlateResult(text="京A12345", color=PlateColor.BLUE, bbox=(50, 50, 100, 40))
        out = draw_result(img, [res])
        assert out.shape == img.shape
        assert not np.array_equal(out, img)
