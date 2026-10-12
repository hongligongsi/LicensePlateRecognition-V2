"""调参 / 评测工具的单元测试。

覆盖三块：
* ``eval_locate`` 的几何原语（坐标口径转换、IoU、贪心匹配、目标函数）
* YOLO 标注读取
* ``tune_io`` 的配置落盘（**必须写进 engine.locate，不能写顶层**）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from eval_locate import (  # noqa: E402
    GRID_FULL,
    GRID_QUICK,
    Report,
    _greedy_match,
    iou_matrix,
    load_labels,
    xywh_to_xyxy,
)
from lpr.config import LocateConfig  # noqa: E402
from tune_io import apply_locate_params, load_existing_locate  # noqa: E402


# --------------------------------------------------------------------------- #
# 坐标口径
# --------------------------------------------------------------------------- #
class TestBboxConversion:
    def test_xywh_becomes_xyxy(self):
        """本项目 bbox 是 (x,y,w,h)，必须转成 (x1,y1,x2,y2) 才能算 IoU。"""
        out = xywh_to_xyxy(np.array([[10, 20, 100, 50]], dtype=np.float32))
        assert out.tolist() == [[10, 20, 110, 70]]

    def test_empty_input(self):
        assert xywh_to_xyxy(np.zeros((0, 4), np.float32)).shape == (0, 4)

    def test_does_not_mutate_input(self):
        """转换必须返回值，不能就地改掉调用方的数组。"""
        src = np.array([[0, 0, 10, 10]], dtype=np.float32)
        _ = xywh_to_xyxy(src)
        assert src.tolist() == [[0, 0, 10, 10]]

    def test_matches_locator_convention(self):
        """真实定位输出 (x,y,w,h) 经过转换后应能算出合理 IoU。"""
        # 定位框紧贴字符带，略小于整牌
        pred_xywh = np.array([[100, 100, 200, 60]], dtype=np.float32)
        gt_xyxy = np.array([[90, 95, 320, 170]], dtype=np.float32)
        iou = iou_matrix(xywh_to_xyxy(pred_xywh), gt_xyxy)[0, 0]
        # 若误把 xywh 当 xyxy 用，x2<x1 会得到 0
        assert 0.3 < iou < 0.95


class TestIoUMatrix:
    def test_identical_boxes(self):
        box = np.array([[0, 0, 10, 10]], dtype=np.float32)
        assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[100, 100, 110, 110]], dtype=np.float32)
        assert iou_matrix(a, b)[0, 0] == 0.0

    def test_half_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[5, 0, 15, 10]], dtype=np.float32)
        # 交集 50，并集 150
        assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150, abs=1e-5)

    def test_shape_with_empty_side(self):
        assert iou_matrix(np.zeros((0, 4), np.float32), np.zeros((3, 4), np.float32)).shape == (
            0,
            3,
        )

    def test_many_to_many_shape(self):
        a = np.zeros((4, 4), np.float32)
        a[:, 2:] = 10
        b = np.zeros((7, 4), np.float32)
        b[:, 2:] = 10
        assert iou_matrix(a, b).shape == (4, 7)


class TestGreedyMatch:
    def test_picks_highest_iou_first(self):
        m = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
        pairs = _greedy_match(m, 0.5)
        assert sorted(pairs) == [(0, 0), (1, 1)]

    def test_respects_threshold(self):
        m = np.array([[0.4]], dtype=np.float32)
        assert _greedy_match(m, 0.5) == []

    def test_one_pred_cannot_match_two_gt(self):
        """贪心必须保证一对一，否则 precision 会被虚高。"""
        m = np.array([[0.9, 0.85]], dtype=np.float32)
        pairs = _greedy_match(m, 0.5)
        assert len(pairs) == 1

    def test_empty_matrix(self):
        assert _greedy_match(np.zeros((0, 0), np.float32), 0.5) == []


# --------------------------------------------------------------------------- #
# 目标函数
# --------------------------------------------------------------------------- #
class TestFitness:
    def test_perfect_report(self):
        # 0.50*R@30 + 0.25*R@50 + 0.25*(1-center_err)
        r = Report(recall30=1.0, recall50=1.0, center_err=0.0)
        assert r.fitness == pytest.approx(1.0)

    def test_partial_report(self):
        r = Report(recall30=0.0, recall50=1.0, center_err=0.0)
        assert r.fitness == pytest.approx(0.50)

    def test_worse_center_error_lowers_fitness(self):
        good = Report(recall30=1.0, recall50=1.0, center_err=0.0).fitness
        bad = Report(recall30=1.0, recall50=1.0, center_err=0.8).fitness
        assert bad < good

    def test_center_error_clamped(self):
        """center_err > 1 时不能让 fitness 变成负数。"""
        r = Report(recall30=1.0, recall50=1.0, center_err=5.0)
        assert r.fitness >= 0.0

    def test_zero_report(self):
        assert Report(center_err=1.0).fitness == pytest.approx(0.0)

    def test_recall30_dominates(self):
        """R@30 权重最高：只改善 R@30 应比只改善 R@50 提升更大。"""
        base = Report(recall30=0.0, recall50=0.0, center_err=1.0).fitness
        only30 = Report(recall30=1.0, recall50=0.0, center_err=1.0).fitness - base
        only50 = Report(recall30=0.0, recall50=1.0, center_err=1.0).fitness - base
        assert only30 > only50


# --------------------------------------------------------------------------- #
# YOLO 标注
# --------------------------------------------------------------------------- #
class TestLoadLabels:
    def test_reads_center_format(self, tmp_path: Path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        boxes = load_labels(p, 200, 100)
        assert boxes.shape == (1, 4)
        assert boxes[0].tolist() == pytest.approx([50.0, 25.0, 150.0, 75.0])

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert load_labels(tmp_path / "nope.txt", 100, 100).shape == (0, 4)

    def test_multiple_lines(self, tmp_path: Path):
        p = tmp_path / "b.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n", encoding="utf-8")
        assert load_labels(p, 100, 100).shape == (2, 4)

    def test_skips_malformed_lines(self, tmp_path: Path):
        p = tmp_path / "c.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\ngarbage\n0 1 2\n", encoding="utf-8")
        assert load_labels(p, 100, 100).shape == (1, 4)

    def test_blank_file(self, tmp_path: Path):
        p = tmp_path / "d.txt"
        p.write_text("", encoding="utf-8")
        assert load_labels(p, 100, 100).shape == (0, 4)


# --------------------------------------------------------------------------- #
# 配置落盘（关键回归：层级必须是 engine.locate）
# --------------------------------------------------------------------------- #
class TestApplyLocateParams:
    def test_writes_into_engine_locate(self, tmp_path: Path):
        (tmp_path / "config.json").write_text(
            json.dumps({"engine": {"locate": {"min_area_ratio": 0.0015}}}),
            encoding="utf-8",
        )
        apply_locate_params({"min_area_ratio": 0.0008}, tmp_path)
        data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert data["engine"]["locate"]["min_area_ratio"] == 0.0008
        # 绝不能出现在顶层
        assert "locate" not in data

    def test_tuple_becomes_list(self, tmp_path: Path):
        apply_locate_params({"morph_kernel_ratio": (0.02, 0.5)}, tmp_path)
        data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert data["engine"]["locate"]["morph_kernel_ratio"] == [0.02, 0.5]

    def test_backup_created_once(self, tmp_path: Path):
        (tmp_path / "config.json").write_text('{"engine": {}}', encoding="utf-8")
        apply_locate_params({"a": 1}, tmp_path)
        bak = tmp_path / "config.json.bak"
        assert bak.exists()
        first = bak.read_text(encoding="utf-8")
        apply_locate_params({"a": 2}, tmp_path)
        # 第二次不得覆盖备份
        assert bak.read_text(encoding="utf-8") == first

    def test_preserves_other_sections(self, tmp_path: Path):
        (tmp_path / "config.json").write_text(
            json.dumps({"engine": {"locate": {}}, "security": {"max_file_mb": 32.0}}),
            encoding="utf-8",
        )
        apply_locate_params({"min_aspect": 2.2}, tmp_path)
        data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert data["security"]["max_file_mb"] == 32.0

    def test_creates_file_when_absent(self, tmp_path: Path):
        apply_locate_params({"max_aspect": 6.0}, tmp_path)
        data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert data["engine"]["locate"]["max_aspect"] == 6.0

    def test_load_existing_roundtrip(self, tmp_path: Path):
        apply_locate_params({"color_margin": 0.06}, tmp_path)
        assert load_existing_locate(tmp_path)["color_margin"] == 0.06

    def test_load_existing_missing_file(self, tmp_path: Path):
        assert load_existing_locate(tmp_path) == {}

    def test_corrupt_config_does_not_crash(self, tmp_path: Path):
        (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
        apply_locate_params({"min_aspect": 2.0}, tmp_path)
        data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert data["engine"]["locate"]["min_aspect"] == 2.0


# --------------------------------------------------------------------------- #
# 搜索网格的健全性
#
# 背景（真实踩过的坑）：网格此前给 morph_kernel_ratio 的候选是 (0.012, 0.30) ~
# (0.025, 0.60)，而该参数的量纲是「相对图宽的比例」，0.60 = 图宽 60% 的形态学核
# —— 那会把整幅图糊成一团。网格还整体偏大、根本没包住默认值 (0.005, 0.021)，
# 于是选出来的"最优"只是网格边缘的退化解。下面这组测试守住这两点。
# --------------------------------------------------------------------------- #
class TestSearchGridSanity:
    def test_grid_brackets_default_values(self):
        """每个搜索维度都必须包住默认值，否则搜索无意义。"""
        base = LocateConfig()
        for name, grid in (("quick", GRID_QUICK), ("full", GRID_FULL)):
            for key, values in grid.items():
                default = getattr(base, key)
                if isinstance(default, tuple):
                    for i, comp in enumerate(default):
                        lo = min(v[i] for v in values)
                        hi = max(v[i] for v in values)
                        assert lo <= comp <= hi, (
                            f"{name} 网格 {key}[{i}] 未包住默认值 {comp}（范围 {lo}~{hi}）"
                        )
                else:
                    assert min(values) <= default <= max(values), (
                        f"{name} 网格 {key} 未包住默认值 {default}"
                    )

    def test_kernel_ratios_are_plausible(self):
        """核比例必须是"小比例"，不能出现 0.5 这种把整图糊掉的量级。"""
        for grid in (GRID_QUICK, GRID_FULL):
            for pair in grid.get("morph_kernel_ratio", ()):
                assert 0 < pair[0] < 0.1, f"核高比例 {pair[0]} 不像核，倒像整幅图"
                assert 0 < pair[1] < 0.15, f"核宽比例 {pair[1]} 不像核，倒像整幅图"

    def test_grid_values_have_correct_types(self):
        """参数类型要和 dataclass 声明一致（box_padding 是 int 像素，不是比例）。"""
        base = LocateConfig()
        for grid in (GRID_QUICK, GRID_FULL):
            for key, values in grid.items():
                default = getattr(base, key)
                if isinstance(default, tuple):
                    assert all(isinstance(v, tuple) and len(v) == len(default) for v in values), (
                        f"{key} 候选必须是等长 tuple"
                    )
                elif isinstance(default, float):
                    assert all(isinstance(v, float) for v in values), f"{key} 必须是 float"
                elif isinstance(default, int):
                    assert all(isinstance(v, int) for v in values), f"{key} 必须是 int"

    def test_grid_does_not_contain_pixel_params(self):
        """像素型参数（如 box_padding）不该出现在比例网格里 —— 容易设成小数。"""
        for grid in (GRID_QUICK, GRID_FULL):
            assert "box_padding" not in grid

    def test_every_grid_key_exists_on_config(self):
        base = LocateConfig()
        for grid in (GRID_QUICK, GRID_FULL):
            for key in grid:
                assert hasattr(base, key), f"网格里的 {key} 不是 LocateConfig 字段"
