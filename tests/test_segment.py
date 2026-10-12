"""引擎注册表与分割模块测试。"""

from __future__ import annotations

import numpy as np
import pytest

from lpr.engines.registry import available_engines, create_engine
from lpr.engines.segment import find_waves, separate_characters


class TestRegistry:
    def test_builtin_registered(self):
        names = available_engines()
        assert "cv" in names
        assert "deep" in names

    def test_unknown_engine_raises(self):
        with pytest.raises(KeyError):
            create_engine("nonexistent")


class TestFindWaves:
    def test_basic_peaks(self):
        hist = np.array([0, 0, 5, 6, 7, 6, 5, 0, 0, 8, 9, 8, 0, 0], dtype=float)
        peaks = find_waves(2.0, hist)
        assert len(peaks) == 2
        assert peaks[0] == (2, 7)
        assert peaks[1] == (9, 12)

    def test_empty(self):
        assert find_waves(1.0, np.array([], dtype=float)) == []

    def test_all_below(self):
        assert find_waves(10.0, np.array([1, 2, 3], dtype=float)) == []


class TestSegmentation:
    def _make_plate(self) -> np.ndarray:
        """合成一张"白底黑字"的假车牌（深色文字 → 需反色场景）。"""
        img = np.full((40, 140), 255, np.uint8)
        # 7 个字符，每个占 16px 宽，间隔 4px
        positions = [4, 24, 44, 64, 84, 104, 124]
        for x in positions:
            img[8:32, x : x + 12] = 0
        return img

    def test_separate_seven_chars(self):
        # 深色文字、浅色底 → 相当于蓝牌场景（需要反色处理）
        # binarize_plate 反色后：黑字变白字
        plate = self._make_plate()
        seg = separate_characters(plate, color_is_dark_text=True)
        assert len(seg.chars) >= 6, f"应切出约 7 个字符，实际 {len(seg.chars)}"

    def test_empty_input(self):
        seg = separate_characters(np.zeros((0, 0), np.uint8), True)
        assert seg.chars == []
