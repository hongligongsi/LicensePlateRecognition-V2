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

import logging
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
    return cv2.bitwise_or(m1, m2)


def _red_signal(roi: np.ndarray) -> float:
    """返回 ROI 内红色像素占比（0~1），并做「字符大小」校验。

    返回值是**过滤后**的红色占比：若最大红色连通域的高度不足 ROI 高度的
    45%，视为散乱反射而非红色字符（真红色字符与车牌等高，
    实测警牌「警」字高占 0.81，尾灯/招牌反射通常 <0.45）。
    """
    bgr = ensure_bgr(roi, min_side=4)
    if bgr is None or bgr.size == 0 or bgr.ndim == 2:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
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
    char_h = stats[idx, cv2.CC_STAT_HEIGHT] / bgr.shape[0]
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
    3. 白色像素显著少于黄色 —— 进一步排除黑底白字牌

    第 3 条是实测后补的：港澳/使领馆等黑底白字牌上，车灯反光、红字渗色
    等会产生零星黄色噪声（占比 4%~8%），足以越过第 2 条的下限，
    但同一块牌上白色文字面积与黄色噪声相当甚至更大（white/yellow≈1~3）；
    真危险品牌上黄字是绝对前景，白色像素≈0，两者可稳定区分。
    """
    bgr = ensure_bgr(roi, min_side=4)
    if not cfg.detect_hazmat or bgr is None or bgr.size == 0:
        return False

    # 灰度图丢失了色相信息，无法区分"黑底黄字"与"黑底白字"，直接放弃判定
    # （调用方应在 BGR 图上判定；灰度输入不会被误判为危险品，只会退回黑牌）
    if bgr.ndim == 2:
        return False

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    total = bgr.shape[0] * bgr.shape[1]
    if total == 0:
        return False

    yellow = cv2.countNonZero(_color_mask(hsv, PlateColor.YELLOW)) / total
    dark = cv2.countNonZero(_color_mask(hsv, PlateColor.BLACK)) / total

    if dark < cfg.hazmat_bg_ratio or yellow < cfg.hazmat_fg_ratio:
        return False
    # 白字排除：白色像素达到黄色的一定倍数 → 前景文字是白色而非黄色
    white = cv2.countNonZero(_color_mask(hsv, PlateColor.WHITE)) / total
    if white >= yellow * cfg.hazmat_white_ratio:
        return False
    # 黄字应只占少数；若黄色多于黑底，说明是黄底黑字的普通黄牌
    return yellow / max(1e-6, yellow + dark) >= cfg.hazmat_fg_share


def _core_region(roi: np.ndarray, trim: float = 0.12) -> np.ndarray:
    """取车牌的中心区域，去掉四周边缘。

    定位框的外圈常混入车身颜色/阴影/相邻物体；中心区域才是稳定的底色。
    裁剪比例不宜过大，否则字符（也占中间）会挤掉底色像素。
    """
    h, w = roi.shape[:2]
    ty, tx = round(h * trim), round(w * trim)
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
    bgr = ensure_bgr(roi, min_side=4)
    if bgr is None or bgr.size == 0:
        return PlateColor.UNKNOWN, 0.0

    core = _core_region(bgr)
    ratios = _color_ratios(core)
    if not ratios:
        return PlateColor.UNKNOWN, 0.0

    # 危险品牌优先判定：黄底黑字的普通黄牌会被后面拦截，
    # 但只有先做极性分析才能把"黑底黄字"从"黄底黑字"里摘出来
    if detect_hazmat(roi, cfg):
        best = max(ratios.get(PlateColor.YELLOW, 0.0), ratios.get(PlateColor.BLACK, 0.0))
        return PlateColor.HAZMAT, max(best, 0.5)

    # ---- 蓝 / 黄 / 绿：相对胜出
    chroma = {c: ratios.get(c, 0.0) for c in (PlateColor.BLUE, PlateColor.YELLOW, PlateColor.GREEN)}
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

    min_share = cfg.color_ratio * 0.25  # 自适应下限（原为 0.5，过严）

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
    """把 4 个角点排成（左上、右上、右下、左下），与输入顺序无关。

    早期实现用「x+y 最小/最大 + y-x 最小/最大」两两配对，只在**侧倾不超过
    45°** 时成立：车牌一旦倾斜更多，配出来的"上边"其实是短边，矫正图会被
    转成 90° 的竖图（实测 220×70 的牌子在 46° 就已颠倒），后面的字符分割
    自然全线失效。

    现在的做法分两步：

    1. **极角环形排序**：按绕质心的极角排序，得到与旋转角无关的稳定环形
       顺序。图像坐标系（y 向下）里这个顺序是顺时针，因此映射到目标矩形
       不会引入镜像翻转；
    2. **挑起点**：4 个环上起点中只有两个能让"上边"是长边（另一个正好是它
       的 180° 反转）。再要求上边**指向右侧**（``tr.x > tl.x``）——车牌从
       左往右阅读，只要侧倾不到 90°，真正的左上角必然满足这一条。

    ``tr.x > tl.x`` 这项剔除了 180° 反转，把可用范围从 45° 扩到 90°；
    这是纯几何能给出的最好结果（正方形般的四边形除外，见下面的打分）。
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ring = pts[np.argsort(angles)]

    best: np.ndarray | None = None
    best_key = (-1.0, False)
    for i in range(4):
        quad = np.roll(ring, -i, axis=0)
        tl, tr, br, bl = quad
        top = float(np.linalg.norm(tr - tl))
        horizontal_pair = top + float(np.linalg.norm(br - bl))
        vertical_pair = float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr))
        if horizontal_pair < vertical_pair:
            continue  # 上边落在短边上 → 会把车牌横竖搞反
        dx = float(tr[0] - tl[0])
        # 先看阅读方向是否朝右，同分时取上边越贴水平越好的那个
        key = (dx / max(top, 1e-6), dx > 0.0)
        if best is None or key > best_key:
            best, best_key = quad, key
    return best if best is not None else ring


def _warp_plate(img: np.ndarray, quad: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """把候选四边形矫正为正向矩形。

    用透视变换而非原版的仿射变换：倾斜拍摄时仿射只能拉正一个方向，
    透视变换能同时处理水平和垂直方向的形变。
    """
    src = _order_quad(quad)
    tl, tr, br, bl = src

    # 目标宽度取上下边的最大值，高度取左右边的最大值
    width = round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width, height = max(width, 1), max(height, 1)

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
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


#: ---- 纵向颜色扩展（"半块双层牌"补救）参数 ----
#: 上下两个方向的探测窗口高度 = bbox 高 × 该值。
#: 取 0.9 的理由：双层牌缺失的半行至多约为已框住半行高度的 60%
#: （双行牌 440x220，上下行同字号），0.9 留出余量又不至于探得太远。
_EXT_WINDOW_RATIO = 0.9
#: 行颜色覆盖率 ≥ 该值视为"牌体在延续"（车牌底色在牌上是主导色）
_EXT_ROW_MIN = 0.35
#: 行颜色覆盖率 < 该值视为"该行有字符"（字符是非底色的前景像素，
#: 会把底色覆盖率咬出一个凹陷——这是"上方还有半行字"的直接证据）
_EXT_CHAR_MAX = 0.85
#: 扩展量低于该像素数不值得重建 ROI（只是边缘毛刺）
_EXT_MIN_PX = 4
#: 最远字符行之上额外保留的边距（含字符顶部与牌框上沿）
_EXT_MARGIN = 7


def _extend_vertical_by_color(
    img: np.ndarray, cand: PlateCandidate, cfg: LocateConfig
) -> PlateCandidate:
    """候选只框住半块双层牌时，把缺失的半行补回来。

    触发场景（实测 img/double_lv.png）：双层牌的上下两行在形态学闭运算
    里没连上（行间距大于纵向闭运算核），轮廓只盖住字符更密的下行
    （140x55、长宽比 3.1 恰好在单行牌区间内，被当成整块单行牌收下），
    上行的"皖11"永远进不了识别。

    做法：在 bbox 正上/正下方各开一个同宽窗口（高 ≤ 0.9×bbox 高），
    从紧贴 bbox 的一端向外做**底色行投影**，扩展量由两条证据共同决定：

    - **颜色延续**：覆盖率 ≥ ``_EXT_ROW_MIN`` 的连续行带才算牌体延续，
      覆盖率跌破阈值的行（牌缘/异色车身）即为止点；
    - **字符证据**：延续带内覆盖率 < ``_EXT_CHAR_MAX`` 的凹陷行。
      只扩到延续带内**最远的凹陷行**（加少量边距）——纯底色的均匀区域
      （蓝车身、树丛、阴影）没有字符凹陷，一行都不扩；
    - **两侧否决**：上下两个方向都满足扩展条件时整块放弃 —— 那是
      候选躺在更大的同色物体内部（黑色引擎盖罩、树冠），而不是
      "半块牌缺了另一半"；真双层牌缺失的半行只在一侧。

    只作用于常规候选（不作用于粘连切半的补救候选 —— 它们本来就是
    从粘连轮廓里切出来的半块，再扩展会重新粘回去）。
    """
    if cand.color is PlateColor.UNKNOWN:
        return cand
    x, y, w, h = cand.bbox
    img_h = img.shape[0]
    # 内缩左右各 6%，避开牌框竖边/斜切残影对行覆盖率的干扰
    inset = max(1, int(w * 0.06))
    x0, x1 = x + inset, x + w - inset
    if x1 <= x0:
        return cand
    win = max(2, int(h * _EXT_WINDOW_RATIO))

    def _scan(y0: int, y1: int, reverse: bool) -> int:
        """算出该方向应扩展的像素数（0 = 不扩展）。"""
        if y1 - y0 < _EXT_MIN_PX:
            return 0
        strip = img[y0:y1, x0:x1]
        mask = _color_mask(cv2.cvtColor(strip, cv2.COLOR_BGR2HSV), cand.color)
        cover = mask.sum(axis=1) // 255 / strip.shape[1]
        if reverse:  # 上方窗口：索引 0 应是紧贴 bbox 的那一行
            cover = cover[::-1]
        # 颜色延续带：从紧贴 bbox 端起覆盖率达标的最长连续行数
        run = 0
        for c in cover:
            if c < _EXT_ROW_MIN:
                break
            run += 1
        if run < _EXT_MIN_PX:
            return 0
        # 延续带里最远的字符凹陷行；没有凹陷 = 均匀底色区，没有可读的字符
        dips = np.nonzero(cover[:run] < _EXT_CHAR_MAX)[0]
        if dips.size == 0:
            return 0
        return min(int(dips.max()) + 1 + _EXT_MARGIN, run)

    up = _scan(max(0, y - win), y, reverse=True)
    down = _scan(y + h, min(img_h, y + h + win), reverse=False)
    if up >= _EXT_MIN_PX and down >= _EXT_MIN_PX:
        return cand  # 两侧都能扩 = 候选在大面积同色物体内部，不是半块牌
    ext = max(up, down)
    if ext < _EXT_MIN_PX:
        return cand

    ny, nh = (y - up, h + up) if up >= down else (y, h + down)
    if nh >= w:  # 扩成竖条必然扩错（车牌恒为横条）
        return cand
    quad = np.array([[x, ny], [x + w, ny], [x + w, ny + nh], [x, ny + nh]], np.float32)
    _crop, bbox = _crop_with_bbox(img, quad, cfg.box_padding)
    if _crop.size == 0:
        return cand
    return PlateCandidate(
        bbox=bbox, roi=_warp_plate(img, quad, bbox), color=cand.color, aspect=w / nh
    )


#: ---- 横向颜色扩展（"尾字被切"补救）参数 ----
#: 左右两个方向的探测窗口宽 = bbox 高 × 该值。
#: 车牌长宽比约 3.2，缺失的尾字段至多约占牌宽 30% ≈ 一个 bbox 高；
#: 1.2 留出余量又不过度外探。
_EXT_H_WINDOW_RATIO = 1.2
#: 两侧扩展量都超过 bbox 宽的该比例时否决：候选躺在更大的同色物体
#: （蓝车身、绿色树丛）内部，而不是"车牌缺了一段"。
_EXT_H_VETO_RATIO = 0.35


def _extend_horizontal_by_color(
    img: np.ndarray, cand: PlateCandidate, cfg: LocateConfig
) -> PlateCandidate:
    """候选只框住车牌前几个字符时，用底色延续把缺失的尾段补回来。

    触发场景（实测 CCPD base 皖A606Y1→皖A606Y、皖AAJ675→皖A·AJ6）：
    末位字符（细笔画 ``1``／两位数字）边缘弱，形态学闭运算连不成块，
    轮廓只盖住前几个字符，尾字整段落在 ROI 之外。这类缺失发生在
    **阅读方向（左右）**，:func:`_extend_vertical_by_color` 补不到。

    做法与纵向扩展对称：在 bbox 左/右侧各开一个窗口（宽 ≤ 1.2×bbox 高）
    做**底色列投影**，扩展到延续带内最远的字符凹陷列（加少量边距）：

    - **颜色延续**：列底色覆盖率 ≥ ``_EXT_ROW_MIN`` 的连续带才算牌体，
      跌破阈值（牌缘/异色车身）即为止点；
    - **字符证据**：只扩到延续带内覆盖率 < ``_EXT_CHAR_MAX`` 的凹陷列；
      纯底色延伸区（蓝车身、绿色树丛）没有凹陷，一列都不扩；
    - **两侧否决**：左右两侧都大幅延续（都 > ``_EXT_H_VETO_RATIO`` × bbox 宽）
      时整块放弃 —— 那是候选躺在更大的同色物体内部，而非车牌缺了一段。

    只作用于常规候选（不作用于粘连切半/双行牌区间候选）。
    绿牌渐变严重，列底色覆盖率本就低且不稳，横向延续带判定不可靠，
    直接跳过（实测对绿牌样张做横向扩展会把已读对的尾字改错）。
    """
    if cand.color in (PlateColor.UNKNOWN, PlateColor.GREEN):
        return cand
    x, y, w, h = cand.bbox
    img_w = img.shape[1]
    # 上下内缩各 8%，避开牌框横边/斜切残影对列覆盖率的干扰
    inset = max(1, int(h * 0.08))
    y0, y1 = y + inset, y + h - inset
    if y1 - y0 < _EXT_MIN_PX:
        return cand
    win = max(2, int(h * _EXT_H_WINDOW_RATIO))

    def _scan(xa: int, xb: int, reverse: bool) -> int:
        """算出该方向应扩展的像素数（0 = 不扩展）。"""
        if xb - xa < _EXT_MIN_PX:
            return 0
        strip = img[y0:y1, xa:xb]
        mask = _color_mask(cv2.cvtColor(strip, cv2.COLOR_BGR2HSV), cand.color)
        cover = mask.sum(axis=0) // 255 / strip.shape[0]
        if reverse:  # 左侧窗口：索引 0 应是紧贴 bbox 的那一列
            cover = cover[::-1]
        # 颜色延续带：从紧贴 bbox 端起覆盖率达标的最长连续列数
        run = 0
        for c in cover:
            if c < _EXT_ROW_MIN:
                break
            run += 1
        if run < _EXT_MIN_PX:
            return 0
        # 延续带里最远的字符凹陷列；没有凹陷 = 均匀底色区，没有可读字符
        dips = np.nonzero(cover[:run] < _EXT_CHAR_MAX)[0]
        if dips.size == 0:
            return 0
        return min(int(dips.max()) + 1 + _EXT_MARGIN, run)

    left = _scan(max(0, x - win), x, reverse=True)
    right = _scan(x + w, min(img_w, x + w + win), reverse=False)
    veto = w * _EXT_H_VETO_RATIO
    if left >= veto and right >= veto:
        return cand  # 两侧都大幅延续 = 候选在更大的同色物体内部
    ext = max(left, right)
    if ext < _EXT_MIN_PX:
        return cand

    nx, nw = (x - left, w + left) if left >= right else (x, w + right)
    if nw > cfg.max_aspect * h:  # 扩得过长必然扩错（车牌长宽比有上限）
        return cand
    quad = np.array([[nx, y], [nx + nw, y], [nx + nw, y + h], [nx, y + h]], np.float32)
    _crop, bbox = _crop_with_bbox(img, quad, cfg.box_padding)
    if _crop.size == 0:
        return cand
    return PlateCandidate(
        bbox=bbox, roi=_warp_plate(img, quad, bbox), color=cand.color, aspect=nw / max(1, h)
    )


def locate_plates(
    img: np.ndarray,
    cfg: LocateConfig,
    logger: logging.Logger | None = None,
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
    # 切半补救的候选单独收集，不与常规候选同池竞争名额（见函数末尾说明）
    salvaged: list[PlateCandidate] = []
    # 双行牌区间（aspect ≤ min_aspect）的候选同样独立成池：
    # 该区间方块状干扰多，同池竞争会挤占常规/补救名额
    double_banded: list[PlateCandidate] = []

    def _candidate_from_box(box: np.ndarray) -> PlateCandidate | None:
        """从 minAreaRect 的角点构造候选；几何不合格返回 ``None``。

        长宽比闸门分两档：
        * 单行牌区间 ``(min_aspect, max_aspect)``：直接通过；
        * 双行牌区间 ``(double_min_aspect, min_aspect]``：双行牌理论
          宽高比 2.0，透视/倾斜后可掉到 2.0 以下（double_yellow.jpg
          实测 1.86）。该区间方块状干扰多，必须额外过"强底色"闸门——
          核心区内蓝/黄/绿单色占比 ≥ ``double_color_ratio``（真双行牌
          实测 ~0.93，方块状车身件/车灯很难达到）。
        """
        side_a = float(np.linalg.norm(box[0] - box[1]))
        side_b = float(np.linalg.norm(box[1] - box[2]))
        long_side, short_side = max(side_a, side_b), min(side_a, side_b)
        if short_side < 1:
            return None
        aspect = long_side / short_side
        if not cfg.double_min_aspect < aspect < cfg.max_aspect:
            return None

        quad = box / scale
        crop, bbox = _crop_with_bbox(img, quad, cfg.box_padding)
        if crop.size == 0:
            return None

        # 用颜色进一步筛掉"长宽比像车牌但不是车牌"的区域（车灯、进气格栅等）
        color, ratio = detect_color(crop, cfg)
        # 透视矫正，得到正向车牌图 —— 后续字符分割依赖它是正的
        warped = _warp_plate(img, quad, bbox)
        # 矫正后重新判一次色：斜切后的裁剪图上颜色噪声更多
        warped_color, warped_ratio = detect_color(warped, cfg)
        if warped_ratio > ratio:
            color, ratio = warped_color, warped_ratio

        # 双行牌区间：强底色闸门（黑/白/未知底色都不可信，只认彩色）
        if aspect <= cfg.min_aspect:
            if color not in (PlateColor.BLUE, PlateColor.YELLOW, PlateColor.GREEN):
                return None
            if ratio < cfg.double_color_ratio:
                return None
        return PlateCandidate(bbox=bbox, roi=warped, color=color, aspect=aspect)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        cand = _candidate_from_box(cv2.boxPoints(cv2.minAreaRect(contour)))
        if cand is not None and cand.aspect > cfg.min_aspect:
            # 尾字补救：颜色还往左右延续的，把缺失的尾段补回来
            cand = _extend_horizontal_by_color(img, cand, cfg)
            # 半块双层牌补救：颜色还往上下延续的，把缺失半行补回来
            candidates.append(_extend_vertical_by_color(img, cand, cfg))
            continue
        if cand is not None:
            # 双行牌区间候选：独立成池、不做纵向扩展（已是完整双行牌
            # 或粘连体，扩展只会把相邻结构再裹进来）。
            # **不 continue**：该轮廓也可能是两块单行牌粘成的矮胖块
            # （hongkang1 粘连体 aspect 1.82 恰落在双行牌区间），
            # 必须继续走下面的粘连补救切半，两条路都留证据。
            double_banded.append(cand)

        # ---- 粘连补救：紧邻的两块车牌（双牌车/教练车/港澳车前后牌）会被
        # 闭运算连成一个矮胖/瘦长轮廓，长宽比掉出车牌区间被整块丢弃
        # （实测 hongkang1.jpg 上下两块牌连成 160x88、aspect 1.82）。
        # 两块牌粘连的形态是确定的：上下粘连 → merged aspect = 单牌/2，
        # 左右粘连 → merged = 单牌×2。仅当 merged 落在"恰好两块牌"的
        # 区间才切，且沿两个轴向各切一半共 4 个半矩形都交给长宽比闸门：
        # 切对的方向恢复单牌比例（160x44→3.6 通过），切错的方向变成
        # 方块（80x88→1.1 被拒），无需预判粘连方向。切错产生的误候选
        # 还要过 detect_color 与 OCR 抽取两道闸门，不会进最终结果。
        area = cv2.contourArea(contour)
        if area >= min_area * 2:
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(contour)
            merged = max(rw, rh) / max(1e-6, min(rw, rh))
            if 1.2 <= merged < 2.05 or 5.45 < merged <= 11.5:
                half_rects = (
                    ((cx - rw / 4, cy), (rw / 2, rh), ang),
                    ((cx + rw / 4, cy), (rw / 2, rh), ang),
                    ((cx, cy - rh / 4), (rw, rh / 2), ang),
                    ((cx, cy + rh / 4), (rw, rh / 2), ang),
                )
                for rect in half_rects:
                    cand = _candidate_from_box(cv2.boxPoints(rect))
                    if cand is not None:
                        salvaged.append(cand)

    # ---- 颜色补充定位：边缘/形态学主链路在模糊、缩放等退化下会丢失
    # 车牌（字符边缘糊掉 → 闭运算连不成块），但**底色是车牌最抗退化的
    # 特征**（double_yellow 全部 6 种退化下黄色掩码仍完整）。
    # 直接对蓝/黄/绿掩码找连通域，过同一套几何闸门后作为补充候选。
    # 独立配额、排在最后：它是兜底召回，不与主链路候选竞争名额。
    color_banded: list[PlateCandidate] = []
    hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    max_area = small.shape[0] * small.shape[1] * 0.05  # 车身级色块直接排除
    ck = ensure_odd(max(3, round(small.shape[1] * 0.02)))  # 闭合字符空洞
    close_kernel = np.ones((ck, ck), np.uint8)
    # 闭运算前先做 3x3 开运算去孤立噪点：噪声退化（如 σ=18 高斯噪声）下
    # 掩码里噪声点密布全图，直接闭运算会把噪声桥接成吞掉真牌的全图巨块
    # （实测 double_yellow noise 下整图连成一个 500x372 连通域，超面积上限被拒）。
    # 3x3 开运算只去孤立点，不伤真牌色块（牌面连续底色远大于 3x3）。
    open_kernel = np.ones((3, 3), np.uint8)
    for mask_color in (PlateColor.BLUE, PlateColor.YELLOW, PlateColor.GREEN):
        mask = _color_mask(hsv_small, mask_color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not min_area <= area <= max_area or w < 1 or h < 1:
                continue
            aspect = w / h
            if not cfg.double_min_aspect < aspect < cfg.max_aspect:
                continue
            quad = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], np.float32) / scale
            crop, bbox = _crop_with_bbox(img, quad, cfg.box_padding)
            if crop.size == 0:
                continue
            color, ratio = detect_color(crop, cfg)
            # 颜色来源的候选必须颜色自洽：掩码颜色与判定颜色一致且够强
            if color is not mask_color or ratio < cfg.double_color_ratio:
                continue
            color_banded.append(PlateCandidate(bbox=bbox, roi=crop, color=color, aspect=aspect))

    # 常规候选与补救候选**分开排序、分开配额**。
    # 为什么补救候选不参与常规名额竞争：切半产生的误候选往往面积巨大
    # （粘连轮廓的一半仍远大于真车牌），同池按面积排序会把真车牌挤出
    # max_candidates 名额——实测 shi_lin_guan（丢沪A1707领）与
    # samples/1.jpg（京E51619→京S1619）两处回归均由此而来。
    # 为什么也不能"只补常规剩余名额"：hongkang1 的真车牌恰是补救候选里
    # 面积较小的下半块，常规候选占满 5 个名额时会再次丢失，故给独立配额。
    # 配额取 2：粘连形态是"恰好两块牌"，4 个半矩形中只有切对方向的
    # 2 个能过长宽比闸门，2 就是补救候选的理论上限。
    key = lambda c: (c.color is PlateColor.UNKNOWN, -c.bbox[2] * c.bbox[3])  # noqa: E731
    candidates.sort(key=key)
    salvaged.sort(key=key)
    double_banded.sort(key=key)
    color_banded.sort(key=key)

    if logger is not None:
        logger.debug(
            "定位完成: 轮廓 %d 个, 候选 %d 个, 补救候选 %d 个, 双行候选 %d 个, 颜色补充 %d 个, 缩放 %.3f",
            len(contours),
            len(candidates),
            len(salvaged),
            len(double_banded),
            len(color_banded),
            scale,
        )

    return (
        candidates[: cfg.max_candidates]
        + salvaged[:2]
        + double_banded[: cfg.double_max_candidates]
        + color_banded[:2],
        scale,
    )


def refine_by_color(roi: np.ndarray, color: PlateColor, cfg: LocateConfig) -> np.ndarray:
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

    # 边界判定用**边缘占比下限**而非纯底色阈值（0.8）：末位字符紧贴牌缘时，
    # 穿过该字符的列底色占比被笔画咬到 0.3~0.7，用 0.8 会把整列当边界裁掉，
    # 尾字整字丢失（实测 CCPD 皖AN9491→皖AN949、皖AV8008→皖A·V800，原始
    # 裁剪本已读对，精定位反而裁掉了尾字）。降到 color_edge_ratio 后，牌外
    # 行列占比≈0 仍被裁掉，牌内带字符的行列全部保留。绿牌渐变严重沿用
    # color_row_ratio_green。
    edge = cfg.color_row_ratio_green if color is PlateColor.GREEN else cfg.color_edge_ratio
    row_thr = cols * edge
    col_thr = rows * edge

    good_rows = np.flatnonzero(row_hits > row_thr)
    good_cols = np.flatnonzero(col_hits > col_thr)

    # 找不到有效行列就保持原样，不要裁成空图
    if good_rows.size < max(3, rows // 6) or good_cols.size < max(3, cols // 6):
        return roi

    y0, y1 = int(good_rows[0]), int(good_rows[-1]) + 1
    x0, x1 = int(good_cols[0]), int(good_cols[-1]) + 1

    cropped = roi[y0:y1, x0:x1]
    # 裁得太狠说明投影不可靠，退回原图。
    # 保留不足 45% 行或 55% 列即视为过度裁剪 —— 正常精定位只修边缘，
    # 切掉一半以上的"投影结果"一定是掩码失效而非真实边界。
    if cropped.shape[0] < rows * 0.45 or cropped.shape[1] < cols * 0.55:
        return roi
    return cropped


__all__ = [
    "PlateCandidate",
    "_order_quad",
    "detect_color",
    "detect_hazmat",
    "locate_plates",
    "refine_by_color",
]
