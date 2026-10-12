"""角点排序（``_order_quad``）的旋转鲁棒性回归。

背景（v2.3.4 修复）：旧实现用「x+y 最小/最大 + y-x 最小/最大」两两配对，
**只在侧倾不超过 45° 时成立**。实测 220×70 的矩形在 46° 就已把"上边"配到
短边上，透视矫正的产物变成 90° 竖图；端到端证据是 ``img/single_blue.jpg``
旋转 50° 后定位**已经命中车牌**（ROI 比例 2.53）却整段漏识——矫正图是竖的，
字符分割自然拿不到东西。

更糟的是当时的 docstring 写着"用叉积判断"，而代码里根本没有叉积，
文档与实现不符让这个 bug 一直没被怀疑。

本文件把「±89° 全范围都能恢复正确顺序」钉死；任何退回 45° 判据的改动
都会在这里红。
"""

from __future__ import annotations

import numpy as np
import pytest

from lpr.engines.locate import _order_quad, _warp_plate

#: 车牌典型比例 440×140 ≈ 3.14:1，取半宽半高便于构造
HALF_W, HALF_H = 110.0, 35.0

#: 局部坐标下的四个角（左上、右上、右下、左下）
_LOCAL = {
    "TL": (-HALF_W, -HALF_H),
    "TR": (HALF_W, -HALF_H),
    "BR": (HALF_W, HALF_H),
    "BL": (-HALF_W, HALF_H),
}


def _rotate(
    deg: float, center: tuple[float, float] = (300.0, 300.0)
) -> dict[str, tuple[float, float]]:
    """按 cv2 的角度约定（正角为逆时针）旋转四个角点。"""
    theta = np.deg2rad(deg)
    cos, sin = np.cos(theta), np.sin(theta)
    return {
        name: (x * cos + y * sin + center[0], -x * sin + y * cos + center[1])
        for name, (x, y) in _LOCAL.items()
    }


def _shuffled(truth: dict[str, tuple[float, float]], seed: int) -> np.ndarray:
    """把真值角点打乱顺序喂进去——排序函数不应依赖输入顺序。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(list(truth))
    return np.float32([truth[k] for k in order])


@pytest.mark.parametrize("deg", range(-89, 90, 1))
def test_recovers_true_corners_at_any_tilt(deg: int):
    """±89° 内任意侧倾、任意输入顺序，都要还原成 左上/右上/右下/左下。"""
    truth = _rotate(deg)
    want = np.float32([truth["TL"], truth["TR"], truth["BR"], truth["BL"]])
    got = _order_quad(_shuffled(truth, seed=abs(deg) + 7))
    assert np.allclose(got, want, atol=1e-3), f"{deg}° 时角点归位错误"


class TestRegressionAgainst45DegreeCliff:
    """旧实现在 46° 就会把车牌横竖搞反，这道坎必须消失。"""

    @pytest.mark.parametrize("deg", [46, 50, 60, 75, 89, -46, -50, -89])
    def test_long_edge_is_horizontal(self, deg: int):
        truth = _rotate(deg)
        tl, tr, _br, bl = _order_quad(_shuffled(truth, seed=1))
        top = float(np.linalg.norm(tr - tl))
        left = float(np.linalg.norm(bl - tl))
        assert top > left, f"{deg}° 时上边仍被配到短边上（会 Warp 成竖图）"

    @pytest.mark.parametrize("deg", [46, 50, 60, 75, 89])
    def test_warp_output_is_landscape(self, deg: int):
        truth = _rotate(deg)
        quad = np.float32([truth[k] for k in ("TL", "TR", "BR", "BL")])
        img = np.full((600, 600, 3), 200, np.uint8)
        warped = _warp_plate(img, quad, (0, 0, 600, 600))
        height, width = warped.shape[:2]
        assert width > height, f"{deg}° 时矫正图是 {height}×{width}（竖的）"

    def test_warp_keeps_plate_aspect(self):
        for deg in (-60, -46, 0, 46, 60, 89):
            truth = _rotate(deg)
            quad = np.float32([truth[k] for k in ("TL", "TR", "BR", "BL")])
            warped = _warp_plate(np.zeros((600, 600, 3), np.uint8), quad, (0, 0, 600, 600))
            height, width = warped.shape[:2]
            assert abs(width / height - 2 * HALF_W / (2 * HALF_H)) < 0.05


class TestOrderIndependence:
    @pytest.mark.parametrize("deg", [0, 17, 44, 46, 63, 88, -35])
    def test_input_permutation_irrelevant(self, deg: int):
        truth = _rotate(deg)
        want = np.float32([truth["TL"], truth["TR"], truth["BR"], truth["BL"]])
        for seed in range(6):
            assert np.allclose(_order_quad(_shuffled(truth, seed=seed)), want, atol=1e-3)

    def test_repeated_calls_are_stable(self):
        truth = _rotate(52)
        quad = _shuffled(truth, seed=3)
        first = _order_quad(quad)
        assert np.allclose(_order_quad(quad), first)


class TestDegenerateInputs:
    def test_axis_aligned_rect(self):
        quad = np.float32([[10, 20], [10, 120], [230, 120], [230, 20]])
        tl, tr, br, bl = _order_quad(quad)
        assert np.allclose(tl, [10, 20])
        assert np.allclose(tr, [230, 20])
        assert np.allclose(br, [230, 120])
        assert np.allclose(bl, [10, 120])

    def test_square_has_no_vertical_warp(self):
        # 正方形没有长短边之分，但输出仍须是一个不自交的环（面积不为 0）
        quad = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
        ordered = _order_quad(quad)
        area = 0.0
        for i in range(4):
            x1, y1 = ordered[i]
            x2, y2 = ordered[(i + 1) % 4]
            area += x1 * y2 - x2 * y1
        assert abs(area) / 2 == pytest.approx(100 * 100)

    def test_all_points_identical_does_not_crash(self):
        quad = np.float32([[5, 5]] * 4)
        assert _order_quad(quad).shape == (4, 2)

    def test_accepts_list_and_flat_array(self):
        as_list = _order_quad([[0, 0], [10, 0], [10, 4], [0, 4]])
        as_flat = _order_quad(np.array([0, 0, 10, 0, 10, 4, 0, 4], dtype=np.float32))
        assert np.allclose(as_list, as_flat)


class TestPerspectiveQuad:
    """不是标准矩形（斜拍梯形）时，也要把最长的上边认作上边。"""

    def test_trapezoid_keeps_long_top(self):
        # 下边比上边略宽（近大远小），整体无明显旋转
        quad = np.float32([[20, 10], [200, 14], [215, 80], [5, 76]])
        tl, tr, _br, bl = _order_quad(quad)
        assert np.allclose(tl, [20, 10], atol=1.5)
        assert np.allclose(tr, [200, 14], atol=1.5)
        assert tr[0] > tl[0]  # 上边指向右侧
        assert np.linalg.norm(tr - tl) > np.linalg.norm(bl - tl)
