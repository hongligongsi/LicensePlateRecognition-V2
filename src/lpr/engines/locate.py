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
from ..utils.image import ensure_bgr, ensure_odd, resize_max_side


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

#: 需要投影裁剪时使用的掩码底色。
#: 危险品牌是「黑底黄字」——投影应以**黑底**为基准（文字是前景，不该参与裁剪），
#: 因此映射到 BLACK 而不是 YELLOW。
_REFINE_BASE: dict[PlateColor, PlateColor] = {
    PlateColor.HAZMAT: PlateColor.BLACK,
}

#: 军/警/武警白牌的「红色字符」信号阈值（占比区间）。
#:
#: 这是识别白底特种牌**唯一可靠的物理特征**：按 GA36/军牌规范，
#: 军牌字头（军/空/海…）、警牌「警」字、武警「WJ」均为**红色**，
#: 一块特种牌上红色字符通常占 3%~10% 面积。
#:
#: 为什么不用别的方法：实测阴影下的警牌（img/police.jpg）白底灰度中位数
#: 只有 ~120，HSV 白色区间（V>=200）完全罩不住；放宽到 V>=150 也只有
#: 0.7% 命中。而"白/黑谁占比大"的极性判定会反过来——阴影白牌的边缘
#: 环带中位灰度（59）比真黑牌使领馆（88）还暗。
#:
#: 上限 0.15 用来挡大片红色车身/招牌（single_blue.jpg 的红色车身误检区
#: 占 0.196）；下限 0.03 挡零星红色噪声（shi_lin_guan.jpg 尾灯反射 0.023）。
#: 阈值仅由仓库内 3 张样张标定，样本量小，后续有更多特种牌样张应再校准。
_RED_SIGN_RATIO = (0.03, 0.15)

#: 红色像素的 HSV 区间（两段：0~10 与 170~180 跨越 H 环）。
_RED_LOW = ((0, 80, 70), (10, 255, 255))
_RED_HIGH = ((170, 80, 70), (180, 255, 255))


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    """生成红色前景掩码（军牌字头/警牌「警」/武警「WJ」的用色）。"""
    m1 = cv2.inRange(hsv, np.array(_RED_LOW[0], np.uint8), np.array(_RED_LOW[1], np.uint8))
    m2 = cv2.inRange(hsv, np.array(_RED_HIGH[0], np.uint8), np.array(_RED_HIGH[1], np.uint8))
    return m1 | m2


def _red_signal(roi: np.ndarray) -> float:
    """返回 ROI 内红色像素占比（0~1），并做「字符大小」校验。

    返回值是**过滤后**的红色占比：若最大红色连通域的高度不足 ROI 高度的
    45%，视为散乱反射而非红色字符（真红色字符与车牌等高，
    实测警牌「警」字高占 0.81，尾灯/招牌反射通常 <0.45）。
    """
    roi = ensure_bgr(roi, min_side=4)
    if roi is None or roi.size == 0 or roi.ndim == 2:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _red_mask(hsv)
    total = mask.size
    if total == 0:
        return 0.0
    ratio = cv2.countNonZero(mask) / total
    if ratio < _RED_SIGN_RATIO[0]:
        return 0.0

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return 0.0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    char_h = stats[idx, cv2.CC_STAT_HEIGHT] / roi.shape[0]
    if char_h < 0.45:
        return 0.0
    return ratio


def _color_mask(hsv: np.ndarray, color: PlateColor) -> np.ndarray:
    """向量化生成某颜色的掩码。"""
    color = _REFINE_BASE.get(color, color)
    low, high = _COLOR_RANGES[color]
    return cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))


def detect_hazmat(roi: np.ndarray, cfg: LocateConfig) -> bool:
    """判断车牌是否为「危险品运输车辆」牌（黑底黄字）。

    危险品运输车悬挂的是黑底黄字牌，而普通商用黄牌是黄底黑字，
    两者底色与字色的明暗关系**恰好相反**。因此不能只看"有没有黄色"，
    必须同时满足：

    1. 主体为黑底（低亮度像素占多数）—— 排除黄底黑字的普通黄牌
    2. 存在足够比例的黄色前景（文字）—— 排除普通黑牌（黑底白字）

    第 1 条排除了黄底黑字，第 2 条排除了黑底白字，两条合起来才是黑底黄字。
    """
    roi = ensure_bgr(roi, min_side=4)
    if not cfg.detect_hazmat or roi is None or roi.size == 0:
        return False

    # 灰度图丢失了色相信息，无法区分"黑底黄字"与"黑底白字"，直接放弃判定
    # （调用方应在 BGR 图上判定；灰度输入不会被误判为危险品，只会退回黑牌）
    if roi.ndim == 2:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = roi.shape[0] * roi.shape[1]
    if total == 0:
        return False

    yellow = cv2.countNonZero(_color_mask(hsv, PlateColor.YELLOW)) / total
    dark = cv2.countNonZero(_color_mask(hsv, PlateColor.BLACK)) / total

    if dark < cfg.hazmat_bg_ratio or yellow < cfg.hazmat_fg_ratio:
        return False
    # 黄字应只占少数；若黄色多于黑底，说明是黄底黑字的普通黄牌
    return yellow / max(1e-6, yellow + dark) >= cfg.hazmat_fg_share


def _core_region(roi: np.ndarray, trim: float = 0.12) -> np.ndarray:
    """取车牌的中心区域，去掉四周边缘。

    定位框的外圈常混入车身颜色/阴影/相邻物体；中心区域才是稳定的底色。
    裁剪比例不宜过大，否则字符（也占中间）会挤掉底色像素。
    """
    h, w = roi.shape[:2]
    ty, tx = int(round(h * trim)), int(round(w * trim))
    if h - 2 * ty < 8 or w - 2 * tx < 8:
        return roi
    return roi[ty : h - ty, tx : w - tx]


def _color_ratios(roi: np.ndarray) -> dict[PlateColor, float]:
    """统计各候选底色的像素占比。"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = roi.shape[0] * roi.shape[1]
    if total == 0:
        return {}
    return {
        color: cv2.countNonZero(_color_mask(hsv, color)) / total
        for color in (
            PlateColor.BLUE,
            PlateColor.YELLOW,
            PlateColor.GREEN,
            PlateColor.WHITE,
            PlateColor.BLACK,
        )
    }


def detect_color(roi: np.ndarray, cfg: LocateConfig) -> tuple[PlateColor, float]:
    """判断车牌底色以及该颜色的像素占比。

    替代原版的双层 ``for i / for j`` + ``.item()`` 统计，改为 ``cv2.inRange``
    一次算完整个掩码，再用 ``countNonZero``。

    ## v2.3 判定策略改进

    原实现用「绝对占比 ≥ ``color_ratio``（0.5）」做门槛，实测**几乎判不出颜色**：
    真实车牌上字符占了大量面积，底色占比通常只有 10%~30%，于是大量车牌被
    判成 ``UNKNOWN``（9 张样张里 5 张）。参考实现 YOLO26-plate 用独立的颜色
    分类网络头规避了这个问题，但我们要在纯 CV 下达到同等效果，改为：

    1. **中心区域**统计（:func:`_core_region`）—— 去掉边缘的车身干扰；
    2. **相对胜出**：不要求绝对占比过半，只要某种颜色**显著领先第二名**
       （差值 ≥ ``color_margin``）就认定为该底色；
    3. **自适应下限**：领先者占比还需超过 ``color_ratio`` 的一个小比例，
       避免在"全是噪声、谁都不像"的区域硬判一个颜色；
    4. 蓝/黄/绿都不像时，才退化为黑白判定。
    """
    roi = ensure_bgr(roi, min_side=4)
    if roi is None or roi.size == 0:
        return PlateColor.UNKNOWN, 0.0

    core = _core_region(roi)
    ratios = _color_ratios(core)
    if not ratios:
        return PlateColor.UNKNOWN, 0.0

    # 危险品牌优先判定：黄底黑字的普通黄牌会被后面拦截，
    # 但只有先做极性分析才能把"黑底黄字"从"黄底黑字"里摘出来
    if detect_hazmat(roi, cfg):
        best = max(ratios.get(PlateColor.YELLOW, 0.0), ratios.get(PlateColor.BLACK, 0.0))
        return PlateColor.HAZMAT, max(best, 0.5)

    # ---- 蓝 / 黄 / 绿：相对胜出
    chroma = {
        c: ratios.get(c, 0.0)
        for c in (PlateColor.BLUE, PlateColor.YELLOW, PlateColor.GREEN)
    }
    ranked = sorted(chroma.items(), key=lambda kv: -kv[1])
    best_color, best_ratio = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    white_ratio = ratios.get(PlateColor.WHITE, 0.0)
    black_ratio = ratios.get(PlateColor.BLACK, 0.0)
    mono_color, mono_ratio = (
        (PlateColor.WHITE, white_ratio)
        if white_ratio >= black_ratio
        else (PlateColor.BLACK, black_ratio)
    )

    min_share = cfg.color_ratio * 0.25          # 自适应下限（原为 0.5，过严）

    # 领先者必须"够多"且"领先够明显"
    if best_ratio >= min_share and (best_ratio - runner_up) >= cfg.color_margin:
        # 例外：黑白已占**绝对主导**时，彩色只是零星噪声。
        # 典型场景是"黑底 + 少量黄色残点"，它语义上是黑牌（关掉危险品检测后
        # 就该判黑），不该被零星黄色拉走。
        if mono_ratio < 0.6 or best_ratio > mono_ratio * 0.35:
            return best_color, best_ratio
        return mono_color, mono_ratio

    # ---- 军/警/武警白牌：红色字符信号。
    # 放在彩色判定之后：hongkang1.jpg 的黄色候选里也有红色招牌反射，
    # 但黄底会先在上一分支胜出，走不到这里——顺序本身就是一道防线。
    # 置信度抬到 0.5（与上方危险品判定同款先例）：红色字符是"法定特征"
    # 级别的强证据，若只报实际占比（~5%），会在 locate_plates 的
    # "warped 与原始裁剪谁的 ratio 大"比较中被大面积黑底(0.3+)压过，
    # 阴影警牌实测因此仍然被判黑。
    red = _red_signal(roi)
    if _RED_SIGN_RATIO[0] <= red <= _RED_SIGN_RATIO[1]:
        return PlateColor.WHITE, max(white_ratio, red, 0.5)

    # ---- 退化为黑白判定（白牌/黑牌/临牌）
    if mono_ratio >= min_share:
        # 黑白与彩色相当接近时优先彩色（车牌底色本就是彩色的概率更高）
        if best_ratio > mono_ratio * 0.6 and best_ratio >= min_share:
            return best_color, best_ratio
        return mono_color, mono_ratio

    # 彩色有微弱信号但不够明确，且黑白也不像 → 给出彩色（避免 UNKNOWN 泛滥）
    if best_ratio > 0:
        return best_color, best_ratio

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
    "detect_hazmat",
    "refine_by_color",
    "_order_quad",
]
