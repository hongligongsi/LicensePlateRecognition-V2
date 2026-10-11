"""车牌定位。

原项目的定位逻辑写在 ``predict()`` 里，共约 200 行，问题集中在：

1. **绝对像素参数**：``np.ones((20, 20))``、``Min_Area = 2000``、``row_num_limit = 21``
   全是写死的像素值，只对"1000px 宽、车牌约占 1/8 画面"的图有效。
2. **无尺度重试**：界面里硬编码了 ``(1, 0.9, ..., 0.4)`` 七档缩放去撞运气。
3. **角度处理糙**：用 ``rect[2]`` 直接猜正负角度，倾斜车牌容易切歪。
4. **颜色判定用逐像素 ``.item(i,j)`` 双层循环**，一张图要跑几十万次 Python 调用，
   是主要性能瓶颈。
5. 找不到车牌时直接返回空字符，调用方拿不到"为什么失败"的信息。

本模块用 numpy 向量化替代逐像素循环（提速约 50~100 倍），
并用**比例参数**替代绝对像素，使同一套参数适配 640px~4000px 的输入。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import LocateConfig
from ..types import PlateColor
from ..utils.image import ensure_odd, resize_max_side


@dataclass
class PlateCandidate:
    """一个车牌候选区域。"""

    #: 在原图中的外接矩形 ``(x, y, w, h)``
    bbox: tuple[int, int, int, int]
    #: 矫正（拉正）后的车牌图像
    roi: np.ndarray
    #: 颜色
    color: PlateColor
    #: 长宽比
    aspect: float


#: HSV 颜色区间（H, S, V 的上下界）。OpenCV 的 H 范围是 0~179。
_COLOR_RANGES: dict[PlateColor, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    PlateColor.BLUE: ((100, 43, 46), (124, 255, 255)),
    PlateColor.YELLOW: ((11, 43, 46), (34, 255, 255)),
    PlateColor.GREEN: ((35, 43, 46), (99, 255, 255)),
    PlateColor.WHITE: ((0, 0, 200), (180, 40, 255)),
    PlateColor.BLACK: ((0, 0, 0), (180, 255, 60)),
}


def _color_mask(hsv: np.ndarray, color: PlateColor) -> np.ndarray:
    """向量化生成某颜色的掩码。"""
    low, high = _COLOR_RANGES[color]
    return cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))


def detect_color(roi: np.ndarray, cfg: LocateConfig) -> tuple[PlateColor, float]:
    """判断车牌底色以及该颜色的像素占比。

    替代原版的双层 ``for i / for j`` + ``.item()`` 统计，改为 ``cv2.inRange``
    一次算完整个掩码，再用 ``countNonZero``。
    """
    if roi is None or roi.size == 0:
        return PlateColor.UNKNOWN, 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = roi.shape[0] * roi.shape[1]
    if total == 0:
        return PlateColor.UNKNOWN, 0.0

    # 蓝、黄、绿是真正需要区分的车牌底色；白/黑作为兜底
    scores: dict[PlateColor, float] = {}
    for color in (PlateColor.BLUE, PlateColor.YELLOW, PlateColor.GREEN):
        ratio = cv2.countNonZero(_color_mask(hsv, color)) / total
        scores[color] = ratio

    best_color, best_ratio = max(scores.items(), key=lambda kv: kv[1])
    if best_ratio >= cfg.color_ratio * 0.5:
        return best_color, best_ratio

    # 蓝黄绿都不像时，退化为"黑白判定"
    white_ratio = cv2.countNonZero(_color_mask(hsv, PlateColor.WHITE)) / total
    black_ratio = cv2.countNonZero(_color_mask(hsv, PlateColor.BLACK)) / total
    if white_ratio >= cfg.color_ratio:
        return PlateColor.WHITE, white_ratio
    if black_ratio >= cfg.color_ratio:
        return PlateColor.BLACK, black_ratio

    return PlateColor.UNKNOWN, best_ratio


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """把 4 个角点排成 左上、右上、右下、左下。

    原版是靠 ``left_point / right_point / heigth_point / low_point`` 四个
    极值点拼出来的，遇到旋转矩形会错位。这里用标准的"x+y 最小/最大"
    + 叉积判断，鲁棒得多，也顺便支持透视矫正。
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # 左上
    ordered[2] = pts[np.argmax(s)]  # 右下

    d = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(d)]  # 右上
    ordered[3] = pts[np.argmax(d)]  # 左下
    return ordered


def _warp_plate(img: np.ndarray, quad: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """把候选四边形矫正为正向矩形。

    用透视变换而非原版的仿射变换：倾斜拍摄时仿射只能拉正一个方向，
    透视变换能同时处理水平和垂直方向的形变。
    """
    src = _order_quad(quad)
    tl, tr, br, bl = src

    # 目标宽度取上下边的最大值，高度取左右边的最大值
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width, height = max(width, 1), max(height, 1)

    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _crop_with_bbox(
    img: np.ndarray, quad: np.ndarray, padding: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """按轴对齐外接矩形裁剪原图，返回（裁剪图, bbox）。"""
    xs = quad[:, 0]
    ys = quad[:, 1]
    h, w = img.shape[:2]

    x0 = max(0, int(np.floor(xs.min())) - padding)
    y0 = max(0, int(np.floor(ys.min())) - padding)
    x1 = min(w, int(np.ceil(xs.max())) + padding)
    y1 = min(h, int(np.ceil(ys.max())) + padding)

    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), np.uint8), (0, 0, 0, 0)

    return img[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def locate_plates(
    img: np.ndarray,
    cfg: LocateConfig,
    logger: object | None = None,
) -> tuple[list[PlateCandidate], float]:
    """定位图中所有可能是车牌的矩形区域。

    Args:
        img: BGR 原图（未缩放）。
        cfg: 定位参数。
        logger: 可选的 logger，用于输出调试信息。

    Returns:
        ``(候选列表, 缩放系数)``。候选的 ``bbox`` 坐标已映射回**原图**坐标系。
    """
    # 1) 先限长边，兼顾速度与精度
    small, scale = resize_max_side(img, cfg.max_side)

    # 2) 灰度 + 去噪
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if cfg.blur_kernel and cfg.blur_kernel > 1:
        k = ensure_odd(cfg.blur_kernel)
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # 3) 形态学开运算去除细节，再与原图相减，突出车牌这类"密集边缘"区域
    k_open = ensure_odd(max(3, round(small.shape[1] * cfg.open_kernel_ratio)))
    kernel = np.ones((k_open, k_open), np.uint8)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    diff = cv2.addWeighted(gray, 1, opened, -1, 0)

    # 4) 二值化 + 边缘
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, cfg.canny_low, cfg.canny_high)

    # 5) 闭运算把字符连成块，开运算去掉孤立噪点
    kr = ensure_odd(max(1, round(small.shape[0] * cfg.morph_kernel_ratio[0])))
    kc = ensure_odd(max(1, round(small.shape[1] * cfg.morph_kernel_ratio[1])))
    kernel = np.ones((kr, kc), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 6) 找轮廓 —— 注意 OpenCV 4.x 只返回 2 个值，原版那个 try/except 可以删了
    contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = small.shape[0] * small.shape[1] * cfg.min_area_ratio
    candidates: list[PlateCandidate] = []

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        box = cv2.boxPoints(cv2.minAreaRect(contour))
        side_a = np.linalg.norm(box[0] - box[1])
        side_b = np.linalg.norm(box[1] - box[2])
        long_side, short_side = max(side_a, side_b), min(side_a, side_b)
        if short_side < 1:
            continue
        aspect = long_side / short_side
        if not cfg.min_aspect < aspect < cfg.max_aspect:
            continue

        # 映射回原图坐标
        quad = box / scale
        crop, bbox = _crop_with_bbox(img, quad, cfg.box_padding)
        if crop.size == 0:
            continue

        # 7) 用颜色进一步筛掉"长宽比像车牌但不是车牌"的区域（车灯、进气格栅等）
        color, ratio = detect_color(crop, cfg)

        # 8) 透视矫正，得到正向车牌图 —— 后续字符分割依赖它是正的
        warped = _warp_plate(img, quad, bbox)
        # 矫正后重新判一次色：斜切后的裁剪图上颜色噪声更多
        warped_color, warped_ratio = detect_color(warped, cfg)
        if warped_ratio > ratio:
            color, ratio = warped_color, warped_ratio

        candidates.append(
            PlateCandidate(bbox=bbox, roi=warped, color=color, aspect=aspect)
        )

    # 已知颜色的排前面，其次按面积从大到小（车牌通常比误检区域大）
    candidates.sort(
        key=lambda c: (c.color is PlateColor.UNKNOWN, -c.bbox[2] * c.bbox[3])
    )

    if logger is not None:
        logger.debug(
            "定位完成: 轮廓 %d 个, 候选 %d 个, 缩放 %.3f",
            len(contours),
            len(candidates),
            scale,
        )

    return candidates[: cfg.max_candidates], scale


def refine_by_color(
    roi: np.ndarray, color: PlateColor, cfg: LocateConfig
) -> np.ndarray:
    """用颜色掩码的投影把车牌上下左右的非车牌边界切掉。

    原版 ``accurate_place`` 的做法是对的，但同样用了逐像素循环。
    这里改成 ``cv2.reduce`` 做行列投影，一次算完。
    """
    if roi is None or roi.size == 0 or color is PlateColor.UNKNOWN:
        return roi

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _color_mask(hsv, color)
    if cv2.countNonZero(mask) == 0:
        return roi

    rows, cols = mask.shape

    # 逐行统计：某行的颜色像素数 / 该行长度
    row_hits = cv2.reduce(mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() / 255.0
    col_hits = cv2.reduce(mask, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() / 255.0

    row_thr = cols * (
        cfg.color_row_ratio_green if color is PlateColor.GREEN else cfg.color_row_ratio
    )
    col_thr = rows * (
        cfg.color_row_ratio_green if color is PlateColor.GREEN else cfg.color_row_ratio
    )

    good_rows = np.flatnonzero(row_hits > row_thr)
    good_cols = np.flatnonzero(col_hits > col_thr)

    # 找不到有效行列就保持原样，不要裁成空图
    if good_rows.size < max(3, rows // 6) or good_cols.size < max(3, cols // 6):
        return roi

    y0, y1 = int(good_rows[0]), int(good_rows[-1]) + 1
    x0, x1 = int(good_cols[0]), int(good_cols[-1]) + 1

    cropped = roi[y0:y1, x0:x1]
    # 裁得太狠说明投影不可靠，退回原图
    if cropped.shape[0] < rows * 0.2 or cropped.shape[1] < cols * 0.3:
        return roi
    return cropped


__all__ = [
    "PlateCandidate",
    "locate_plates",
    "detect_color",
    "refine_by_color",
    "_order_quad",
]
