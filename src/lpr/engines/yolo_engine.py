"""车牌专用检测 + 识别引擎（ONNX，第三引擎）。

**定位**：补上「传统 CV 定位」的短板。``cv`` / ``deep`` 两个引擎的车牌*定位*
都依赖 OpenCV 形态学轮廓，对**侧倾、模糊、低对比度**车牌召回不足
（见 ``docs/REFERENCE_REVIEW.md`` 第 231 行：「侧倾车牌的定位召回是下一个短板」）。
本引擎用**可训练的检测网络**（YOLO-pose 思路：一次前向给出 bbox + 4 角点 +
单/双层类别）替代轮廓定位，角点直接供透视矫正使用。

**与 ``deep`` 的关系**：两者互补而非替代。
``deep`` = CV 定位引导 + 纯识别（快，依赖定位成功）；
``yolo`` = 学习式检测 + 识别（慢，但定位鲁棒）。

**为什么只依赖 onnxruntime**：不引入 torch。权重由用户按需下载或以
Release 附件形式提供，仓库不捆绑（见 ``docs/THIRD_PARTY.md`` 与 README 许可一节）。
这样安装体积保持在「几十 MB 的 onnxruntime」，而不是「2GB+ 的 PyTorch」。

**许可证提示**：若使用基于 Ultralytics YOLO 训练的权重，该权重受
**AGPL-3.0** 约束。本项目自身为 GPL-3.0；以此为第三引擎时请确认你的
使用场景是否满足 AGPL 的源码提供义务（尤其是「作为网络服务提供」的情形）。

## 推理管线

```
输入图
  │
  ├─(1) letterbox 到 640×640（保持比例 + 灰边 114）   ← _letterbox
  │
  ├─(2) 一次前向 → 输出张量                          ← _infer
  │
  ├─(3) 解析张量布局 → bbox / 类别 / 4 角点           ← resolve_output_spec
  │       · 支持 v8（channels-first）与 v5（anchors-first）
  │       · 支持有/无 objectness 分支、角点维度 2 或 3
  │       · 无角点的纯检测权重也能跑（角点退化为外接框）
  │
  ├─(4) 置信度阈值 + 按类 NMS                        ← nms
  │
  ├─(5) 角点逆 letterbox → 透视矫正成正向矩形         ← _warp
  │       （复用 locate 的 _order_quad，侧倾 >45° 也不会把牌照转成竖图）
  │
  └─(6) 识别字符
          · 有 plate_rec.onnx → 内置 CTC 解码        ← ctc_decode
          · 只有检测权重     → 复用 deep 引擎的识别管线
                               （多尺度投票 + 语法纠错 + 红字尾补）
```

第 (6) 步**刻意复用** ``deep`` 引擎的识别管线而不是另写一份：车牌识别里
真正的难点（多尺度投票、语法纠错、警/领红字尾补）都在那里，抄一份出来
必然会随两边改动而漂移。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import EngineConfig, LocateConfig
from ..plate_rules import correct, plate_score
from ..types import PlateColor, PlateLayout, PlateResult
from ..utils.image import imread
from ..utils.logging_setup import get_logger
from ..utils.security import Limits, check_image
from .base import BaseEngine, EngineUnavailable
from .locate import _order_quad, _warp_plate

logger = get_logger("engines.yolo")

#: 本引擎默认查找权重的相对目录（相对项目根）
DEFAULT_MODEL_DIR = Path("models") / "yolo"

#: 检测模型默认文件名
DETECT_MODEL_NAME = "plate_det.onnx"
#: 识别模型默认文件名（可选；缺省时复用 RapidOCR 做识别）
RECOGNIZE_MODEL_NAME = "plate_rec.onnx"
#: 识别模型配套字符表（每行一个字符，**第 1 行是 CTC blank 占位**）
CHARSET_NAME = "charset.txt"

#: 输入尺寸：YOLO 系惯用的 640；改这个必须与导出时的 imgsz 一致
DEFAULT_INPUT_SIZE = 640

#: 检测置信度阈值
DEFAULT_CONF_THRESHOLD = 0.25
#: NMS IoU 阈值
DEFAULT_IOU_THRESHOLD = 0.45

#: letterbox 灰边填充值（Ultralytics 默认 114）
LETTERBOX_VALUE = 114

#: 车牌在**矫正后**的合理宽高比区间。
#: 单行牌 440×140mm ≈ 3.14:1，双层牌 440×220mm ≈ 2.0:1。
#: 留足透视误差余量：超出这个范围的检测框基本是车身/灯具/格栅的误检。
_MIN_ASPECT = 1.2
_MAX_ASPECT = 8.0

#: 矫正后车牌的最小高度（像素）。低于它的检测框送去识别只会浪费算力。
_MIN_WARPED_HEIGHT = 8

#: 识别网络（CRNN）输入规格 —— 必须与导出时一致
REC_INPUT_HEIGHT = 48
REC_MAX_WIDTH = 320
#: CTC blank 在字符表里的下标（约定第 0 位）
CTC_BLANK_INDEX = 0

#: 属性列顺序约定 A：``box(4) + [obj(1)] + cls(nc) + kpts(4×2)``
#: —— Ultralytics YOLOv8/v5-pose 导出（``cls`` 在关键点之前），主流写法。
ATTR_ORDER_CLS_FIRST = "cls_first"

#: 属性列顺序约定 B：``box(4) + obj(1) + kpts(4×2) + cls(nc)``
#: —— ``we0091234/Chinese_license_plate_detection_recognition``（GPL-3.0）
#: 的 ``plate_detect.onnx`` 就是这个顺序：它的后处理是
#: ``dets[:, :, 4]`` 取 obj、``dets[:, 5:13]`` 取四角点、``dets[:, 13:15]``
#: 取两类的 argmax。
ATTR_ORDER_KPT_FIRST = "kpt_first"

#: 受支持的属性列顺序
ATTR_ORDERS = (ATTR_ORDER_CLS_FIRST, ATTR_ORDER_KPT_FIRST)


def _round_int(value: float) -> int:
    """四舍五入取整，并保证返回 **Python int**（不泄漏 numpy 标量）。

    刻意不写成 ``int(round(x))``：``round()`` 的返回已声明为 ``int``，
    再套一层 ``int()`` 会被 ruff 的 RUF046 判成多余转换。先 ``float()``
    是为了让 numpy 标量在 ``round()`` 前就转成 Python 浮点 —— 这样
    bbox 元组里不会混进 ``np.float32``，``to_dict()`` 序列化才不会炸。
    """
    return round(float(value))


def _probe_onnxruntime() -> tuple[bool, str]:
    """检查 onnxruntime 是否可用（本引擎唯一的硬依赖）。"""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False, "未安装 onnxruntime: pip install onnxruntime"
    return True, ""


def _missing_model_message(model: Path) -> str:
    """权重缺失时的提示文案（类方法与实例方法共用，免得两处说法不一致）。"""
    return (
        f"未找到检测模型 {model}；"
        '下载权重后放到该路径即可启用（安装 extras: pip install -e ".[yolo]"；'
        "或用 python tools/fetch_yolo_weights.py 校验后落盘；"
        "只想先验证管道可用，可先用 python tools/make_stub_onnx.py 生成桩模型）"
    )


def locate_model_dir(models_dir: Path | None = None) -> Path:
    """定位权重目录。

    优先用调用方显式传入的 ``models_dir``；否则从本文件上溯到项目根，
    再拼 ``models/yolo``。
    """
    if models_dir is not None:
        return models_dir
    # src/lpr/engines/yolo_engine.py -> 上溯 4 层到项目根
    root = Path(__file__).resolve().parents[3]
    return root / DEFAULT_MODEL_DIR


# ============================================================ 输出布局解析


@dataclass(frozen=True)
class OutputSpec:
    """模型输出张量的语义布局。

    不同 YOLO 版本导出的 pose 模型，输出维度含义并不一致。与其"猜一个
    常见形状"，不如把**能解析出来的布局显式记录下来**：解码时按它取列，
    日志里也能打印出来，接错权重时一眼看出是形状对不上还是列错位。
    """

    #: True = ``(1, C, N)``（Ultralytics v8 系）；False = ``(1, N, C)``（v5 系）
    channels_first: bool
    #: 是否存在独立的 objectness 分支（v5 有、v8 无）
    has_objectness: bool
    #: 角点个数（0 = 该权重只有检测头，没有关键点分支）
    num_points: int
    #: 每个角点的维度（2 = (x,y)；3 = (x,y,conf)）
    point_dim: int
    #: 类别数
    num_classes: int
    #: 属性列顺序，见 :data:`ATTR_ORDER_CLS_FIRST` / :data:`ATTR_ORDER_KPT_FIRST`
    attr_order: str = ATTR_ORDER_CLS_FIRST

    @property
    def attr_count(self) -> int:
        """单条预测的属性数：``4 + obj + nc + npts * kdim``。

        **注意**：这个式子与列顺序无关 —— 两种顺序的属性数完全相同。
        因此仅凭形状无法区分 A/B 两种约定，``attr_order`` 必须由调用方指定
        （或按权重来源推断），见 :attr:`attr_order`。
        """
        return 4 + int(self.has_objectness) + self.num_classes + self.num_points * self.point_dim

    @property
    def class_offset(self) -> int:
        """类别分数在属性维里的起始下标。"""
        if self.attr_order == ATTR_ORDER_KPT_FIRST:
            # box | obj | kpts | cls
            return 4 + int(self.has_objectness) + self.num_points * self.point_dim
        # box | [obj] | cls | kpts
        return 4 + int(self.has_objectness)

    @property
    def points_offset(self) -> int:
        """四角点在属性维里的起始下标。"""
        if self.attr_order == ATTR_ORDER_KPT_FIRST:
            # box | obj | kpts | cls
            return 4 + int(self.has_objectness)
        # box | [obj] | cls | kpts
        return 4 + int(self.has_objectness) + self.num_classes


def _match_attr_count(
    attr: int,
    num_classes: int,
    point_dim: int | None,
    has_objectness: bool | None,
) -> tuple[bool, int, int] | None:
    """在给定属性数下反解 ``(has_objectness, num_points, point_dim)``。

    枚举顺序刻意写死：先试无 objectness（v8 系更常见），再试有；
    角点维度先试 2（CCPD 导出的 pose 权重就是 ``kpt_shape: [4,2]``）再试 3。
    这样同一属性数下若有多种解释，命中的是最常见的那个。
    """
    for has_obj in (has_objectness,) if has_objectness is not None else (False, True):
        for kdim in (point_dim,) if point_dim is not None else (2, 3):
            for npts in (4, 0):
                if 4 + int(has_obj) + num_classes + npts * kdim == attr:
                    return has_obj, npts, kdim
    return None


def resolve_output_spec(
    shape: tuple[int, ...] | list[int],
    *,
    num_classes: int = 2,
    point_dim: int | None = None,
    has_objectness: bool | None = None,
    attr_order: str = ATTR_ORDER_CLS_FIRST,
) -> OutputSpec:
    """把 ONNX 输出张量的形状解析成 :class:`OutputSpec`。

    只有**两个非 1 维度**（batch 维被忽略）才算合法预测张量；其中较小的
    那一维几乎必然是属性数（锚点数动辄 8400 / 25200，属性数一般不超过
    几十），据此先定朝向，再用 :func:`_match_attr_count` 验证属性数能否
    被已知的语义组合整除；不行就换另一个朝向再试。

    Args:
        attr_order: 属性列顺序。**形状无法区分这件事** —— 例如
            ``box+obj+cls+kpts`` 与 ``box+obj+kpts+cls`` 都是 15 个属性。
            默认按 Ultralytics 的 ``cls_first``；用 ``we0091234`` 系权重
            （``plate_detect.onnx``）时必须显式传 ``kpt_first``，否则会把
            角点当类别读、类别当角点读，结果是一堆看似合理的错框。

    Raises:
        ValueError: 形状非法，或属性数无法用任何受支持的布局解释。
            错误信息里带上形状与已尝试的假设 —— 接错权重时这是唯一线索。
    """
    if attr_order not in ATTR_ORDERS:
        raise ValueError(f"未知属性列顺序 {attr_order!r}，可选: {', '.join(ATTR_ORDERS)}")

    dims = [int(d) for d in shape if int(d) != 1]
    if len(dims) != 2:
        raise ValueError(
            f"检测模型输出应为 2 维（不含 batch 维），实际形状 {tuple(shape)}；"
            "若是 (1, 3, H, W) 这类输入张量，说明取错了输出索引"
        )
    a, b = dims
    # 属性数 ≤ 锚点数：先按这个假设定朝向，失败再反过来
    order = (True, False) if a <= b else (False, True)
    tried: list[str] = []
    for channels_first in order:
        attr = a if channels_first else b
        hit = _match_attr_count(attr, num_classes, point_dim, has_objectness)
        tried.append(f"{'C×N' if channels_first else 'N×C'} attr={attr}")
        if hit is not None:
            has_obj, npts, kdim = hit
            return OutputSpec(channels_first, has_obj, npts, kdim, num_classes, attr_order)

    raise ValueError(
        f"无法解析检测模型输出形状 {tuple(shape)}（已试 {', '.join(tried)}）。"
        f"请用 num_classes / point_dim / has_objectness 显式指明布局；"
        f"当前 num_classes={num_classes}, point_dim={point_dim}, "
        f"has_objectness={has_objectness}"
    )


def decode_predictions(
    pred: np.ndarray, spec: OutputSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """把原始预测张量解成 ``(boxes_xyxy, scores, class_ids, points)``。

    坐标系**仍是 letterbox 之后的输入图坐标系**，调用方负责逆变换回原图。

    Args:
        pred: 原始输出张量（含或不含 batch 维）。
        spec: 由 :func:`resolve_output_spec` 得到的布局说明。

    Returns:
        ``boxes_xyxy`` ``(N, 4)``、``scores`` ``(N,)``、``class_ids`` ``(N,)``、
        ``points`` ``(N, 4, 2)`` 或 ``None``（权重无关键点分支时）。

    Notes:
        置信度的算法刻意与 Ultralytics 对齐：**有** objectness 时
        ``conf = obj × max(cls)``，**无** objectness 时 ``conf = max(cls)``。
        写成 ``obj × cls``（不过 max）会让同一张图在不同权重下阈值行为不一致。

        类别列与角点列的位置由 ``spec.attr_order`` 决定 —— 见
        :data:`ATTR_ORDER_CLS_FIRST` / :data:`ATTR_ORDER_KPT_FIRST`。
    """
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"仅支持 batch=1 的检测输出，实际 batch={arr.shape[0]}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"检测输出应为 2 维，实际 {arr.ndim} 维")
    if spec.channels_first:
        arr = arr.T  # (attr, N) -> (N, attr)

    n = arr.shape[0]
    if arr.shape[1] < spec.attr_count:
        raise ValueError(
            f"属性维长度 {arr.shape[1]} 小于布局要求的 {spec.attr_count}；输出布局与权重不匹配"
        )

    centers = arr[:, :4]
    cls_off, kpt_off = spec.class_offset, spec.points_offset
    class_scores = arr[:, cls_off : cls_off + spec.num_classes]
    class_ids = class_scores.argmax(axis=1).astype(np.int32)
    best = class_scores[np.arange(n), class_ids]
    if spec.has_objectness:
        best = best * arr[:, 4]

    cx, cy, bw, bh = centers[:, 0], centers[:, 1], centers[:, 2], centers[:, 3]
    boxes = np.stack([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], axis=1).astype(
        np.float32
    )

    points: np.ndarray | None = None
    if spec.num_points:
        span = spec.num_points * spec.point_dim
        tail = arr[:, kpt_off : kpt_off + span]
        points = tail.reshape(n, spec.num_points, spec.point_dim)[:, :, :2].astype(np.float32)

    return boxes, best.astype(np.float32), class_ids, points


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    class_ids: np.ndarray | None = None,
) -> list[int]:
    """非极大值抑制，返回保留框的下标（按分数从高到低）。

    ``class_ids`` 非空时按类**分别**抑制：单行牌与双层牌是不同类别，
    一张图里两者重叠（如挂车前后牌）时不该互相把对方压掉。
    """
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []

    if class_ids is None:
        groups = [order]
    else:
        groups = [order[class_ids[order] == c] for c in np.unique(class_ids[order])]

    for group in groups:
        remaining = list(group)
        while remaining:
            best = remaining.pop(0)
            keep.append(int(best))
            if not remaining:
                continue
            rest = np.asarray(remaining, dtype=np.int64)
            ious = _iou_matrix(boxes[best], boxes[rest])
            remaining = [int(i) for i, v in zip(remaining, ious, strict=True) if v <= iou_threshold]

    keep.sort(key=lambda i: -float(scores[i]))
    return keep


def _iou_matrix(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """单个框与一批框的 IoU。"""
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return np.asarray(out, dtype=np.float32)


def letterbox(
    img: np.ndarray, size: int, value: int = LETTERBOX_VALUE
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比缩放到 ``size×size`` 并居中填边。

    四边填充量用 ``round(x - 0.1) / round(x + 0.1)`` 而不是各自取整：
    这样上下（左右）填充之和恒等于 ``size - 缩放后边长``，不会因为两次
    独立取整而少 1 像素、导致坐标整体偏移半个像素。
    该口径与 Ultralytics 的 ``letterbox`` 一致 —— 与训练侧对齐比"更正确"
    的取整更重要。

    Returns:
        ``(填充后的图, 缩放比, (左填充, 上填充))``
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"图像尺寸非法: {img.shape}")
    ratio = min(size / h, size / w)
    new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (size - new_w) / 2
    pad_h = (size - new_h) / 2
    top, left = _round_int(pad_h - 0.1), _round_int(pad_w - 0.1)
    bottom, right = _round_int(pad_h + 0.1), _round_int(pad_w + 0.1)
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(value, value, value)
    )
    return out, float(ratio), (left, top)


def ctc_decode(logits: np.ndarray, charset: list[str]) -> tuple[str, list[float], float]:
    """CTC 贪心解码。

    Args:
        logits: 识别网络输出，``(T, C)`` 或 ``(1, T, C)``（T = 时间步，C = 字符表大小）。
        charset: 字符表，**下标 0 必须是 CTC blank**。

    Returns:
        ``(文本, 逐字符置信度, 整体置信度)``

    Notes:
        三步标准流程：argmax → 合并相邻重复 → 去掉 blank。
        ``merge_repeated`` 必须在**去 blank 之前**做，否则 ``a a`` 这种
        真实重复字符会被错误合并（CTC 里相邻同字符本来就要靠中间的
        blank 分隔，直接先去 blank 会丢字）。

        ``(T, C)`` 与 ``(C, T)`` 的判别**用字符表长度**，不是比大小：
        字符表是已知量（70 上下），而时间步长 T 可能比它小得多（很短的车牌
        被裁得很紧时 T 只有几个时间步）。早先按"谁小谁是字符表"来猜，
        遇到 T < C 就会把矩阵转错、整串读成乱码。
    """
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"仅支持 batch=1 的识别输出，实际 batch={arr.shape[0]}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"识别输出应为 2 维，实际 {arr.ndim} 维")

    n_chars = len(charset)
    if arr.shape[1] == n_chars:
        pass  # 已是 (T, C)
    elif arr.shape[0] == n_chars:
        arr = arr.T
    else:
        raise ValueError(
            f"识别输出形状 {tuple(arr.shape)} 与字符表长度 {n_chars} 不匹配："
            "两者必须有一维等于字符表长度（字符表第 1 行是 CTC blank 占位）"
        )

    shifted = arr - arr.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)

    indices = probs.argmax(axis=1)
    chars: list[str] = []
    confs: list[float] = []
    prev = CTC_BLANK_INDEX
    for t, idx in enumerate(indices):
        cur = int(idx)
        if cur != prev and cur != CTC_BLANK_INDEX and 0 <= cur < len(charset):
            chars.append(charset[cur])
            confs.append(float(probs[t, cur]))
        prev = cur

    text = "".join(chars)
    overall = float(np.mean(confs)) if confs else 0.0
    return text, confs, overall


# ============================================================ 数据结构


@dataclass(frozen=True)
class Detection:
    """一次检测输出：车牌框、四角点、版式类别、置信度。"""

    #: 四角点，顺序为 左上 / 右上 / 右下 / 左下，形状 (4, 2)，float32
    corners: np.ndarray
    #: 轴对齐外接框 (x1, y1, x2, y2)
    box: tuple[float, float, float, float]
    #: 单层 / 双层
    layout: PlateLayout
    #: 检测置信度
    score: float
    #: 分类下标（调试用：接错权重时能看出类别映射是否反过来）
    class_id: int = 0

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


# ============================================================ 引擎


class YoloEngine(BaseEngine):
    """车牌专用检测 + 识别引擎（ONNX 推理）。

    检测权重是**必需**的（``models/yolo/plate_det.onnx``）；识别权重可选：

    * 有 ``plate_rec.onnx`` + ``charset.txt`` → 用内置 CTC 解码识别；
    * 没有 → 复用 ``deep`` 引擎的识别管线（多尺度投票 + 语法纠错）；
    * 连 ``deep`` 也不可用 → **纯检测模式**：返回带框但 ``text`` 为空的结果。
      这不是失败，是明确的降级：``tools/eval_ccpd.py`` 之类的定位评测
      只需要框，静默返回空列表反而让人以为模型坏了。
    """

    name = "yolo"
    display_name = "检测识别引擎"
    description = (
        "车牌专用检测网络（bbox + 四角点 + 单/双层）+ 识别，"
        "对侧倾/模糊车牌的定位召回优于轮廓法；需自行下载 ONNX 权重"
    )

    def __init__(
        self,
        config: Any = None,
        models_dir: Path | None = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        providers: list[str] | None = None,
        *,
        num_classes: int = 2,
        point_dim: int | None = None,
        has_objectness: bool | None = None,
        attr_order: str | None = None,
        double_class_ids: tuple[int, ...] = (1,),
        rgb: bool = True,
        recognize_backend: str | None = None,
    ) -> None:
        """
        Args:
            config: ``EngineConfig``（用于取 ``locate`` 参数与识别引擎配置）。
            models_dir: 权重目录，默认 ``<项目根>/models/yolo``。
            input_size: 与导出时的 ``imgsz`` **必须一致**。
            num_classes: 检测类别数。
            point_dim: 角点维度；``None`` = 由输出形状自动判定。
            has_objectness: 是否有独立 objectness 分支；``None`` = 自动判定。
            attr_order: 属性列顺序（``cls_first`` / ``kpt_first``）。
                **形状区分不出来**，用 ``we0091234`` 系权重时要显式改成
                ``kpt_first``。
            double_class_ids: 哪些类别下标代表**双层牌**。
            rgb: 输入是否转 RGB。Ultralytics 训练用 RGB，默认 True。
            recognize_backend: ``auto`` / ``crnn`` / ``deep`` / ``none``。
        """
        super().__init__()
        self.config = config
        #: 参数解析优先级：**显式实参 > config（yolo_* 字段）> 引擎默认值**。
        #: config 来源见 ``EngineConfig.yolo_attr_order`` 等（GUI/CLI 走这条路）。
        cfg_attr_order = getattr(config, "yolo_attr_order", "") if config else ""
        cfg_backend = getattr(config, "yolo_recognize_backend", "") if config else ""
        cfg_models_dir = getattr(config, "yolo_models_dir", "") if config else ""
        self.models_dir = locate_model_dir(
            models_dir
            if models_dir is not None
            else (Path(cfg_models_dir) if cfg_models_dir else None)
        )
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        #: None = 交给 onnxruntime 自动选（有 CUDA 会用 CUDAExecutionProvider）
        self.providers = providers
        self.num_classes = num_classes
        self.point_dim = point_dim
        self.has_objectness = has_objectness
        self.attr_order = (
            attr_order if attr_order is not None else (cfg_attr_order or ATTR_ORDER_CLS_FIRST)
        )
        self.double_class_ids = tuple(double_class_ids)
        self.rgb = rgb
        self.recognize_backend = (
            recognize_backend if recognize_backend is not None else (cfg_backend or "auto")
        )

        self._detect_session: Any = None
        self._recognize_session: Any = None
        self._input_name = ""
        self._rec_input_name = ""
        self._charset: list[str] = []
        self._provider_used = ""
        self._spec: OutputSpec | None = None
        #: 识别回退：复用 deep 引擎的识别管线（惰性创建）
        self._recognizer_engine: Any = None
        self._recognizer_failed = False

    # -------------------------------------------------------------- 路径

    @property
    def detect_model_path(self) -> Path:
        return self.models_dir / DETECT_MODEL_NAME

    @property
    def recognize_model_path(self) -> Path:
        return self.models_dir / RECOGNIZE_MODEL_NAME

    @property
    def charset_path(self) -> Path:
        return self.models_dir / CHARSET_NAME

    # -------------------------------------------------------------- 可用性

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """仅在 onnxruntime 与**默认目录下的**权重都具备时才算可用。

        这是给注册表/界面用的**类方法**（``describe_engines`` 只能这么调），
        因此它只看 ``models/yolo/``，看不到某个实例自定义的 ``models_dir``。
        实例自己该用 :meth:`_check_available` —— 否则把权重放在别处的用户
        会遇到"文件明明在、却报找不到"（这个坑真的踩到了：桩模型自检时
        指向临时目录，加载却仍在找默认路径）。
        """
        return cls._probe_path(locate_model_dir() / DETECT_MODEL_NAME)

    @staticmethod
    def _probe_path(model: Path) -> tuple[bool, str]:
        """给定检测权重路径，判断是否可用。"""
        ok, msg = _probe_onnxruntime()
        if not ok:
            return False, msg
        if not model.is_file():
            return False, _missing_model_message(model)
        return True, ""

    def _check_available(self) -> tuple[bool, str]:
        """按**本实例**的权重路径判断可用性。"""
        return self._probe_path(self.detect_model_path)

    # -------------------------------------------------------------- 加载

    def load(self) -> None:
        ok, reason = self._check_available()
        if not ok:
            raise EngineUnavailable(reason)

        import onnxruntime as ort

        opts = ort.SessionOptions()
        #: 与 CI / 批量识别共存：限制线程数，避免与多进程批量互相抢核
        opts.intra_op_num_threads = 0  # 0 = 交给 onnxruntime 自行决定
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = self.providers or ort.get_available_providers()
        try:
            self._detect_session = ort.InferenceSession(
                str(self.detect_model_path), sess_options=opts, providers=providers
            )
        except Exception as exc:
            raise EngineUnavailable(
                f"检测模型加载失败: {self.detect_model_path}（{type(exc).__name__}: {exc}）"
            ) from exc

        self._input_name = self._detect_session.get_inputs()[0].name
        self._provider_used = self._detect_session.get_providers()[0]

        # 识别模型可选：缺失时复用 deep 引擎的 RapidOCR 做字符识别
        self._recognize_session = None
        self._charset = []
        if self.recognize_backend in ("auto", "crnn") and self.recognize_model_path.is_file():
            self._load_recognize_model(ort, opts, providers)

        logger.info(
            "检测识别引擎就绪 | 检测=%s | 识别=%s | 后端=%s | 输入=%d",
            self.detect_model_path.name,
            self._recognize_label(),
            self._provider_used,
            self.input_size,
        )
        self._ready = True

    def _recognize_label(self) -> str:
        """日志里显示的识别后端名。"""
        if self._recognize_session is not None:
            return f"{self.recognize_model_path.name}({len(self._charset)} 字表)"
        if self.recognize_backend == "none":
            return "关闭（纯检测）"
        if self.recognize_backend == "crnn":
            return "未加载（缺识别权重或字符表）"
        return "复用 deep 引擎"

    def _load_recognize_model(self, ort: Any, opts: Any, providers: list[str]) -> None:
        """加载识别权重与字符表；任何失败都降级而不是抛异常。

        识别是**可选**能力：一个坏掉的 ``plate_rec.onnx`` 不该让整个检测
        引擎不可用（那是用户能感知到的功能倒退）。
        """
        try:
            self._recognize_session = ort.InferenceSession(
                str(self.recognize_model_path), sess_options=opts, providers=providers
            )
            self._rec_input_name = self._recognize_session.get_inputs()[0].name
        except Exception:
            logger.warning(
                "识别模型加载失败，将回退到复用 deep 引擎识别: %s",
                self.recognize_model_path,
                exc_info=True,
            )
            self._recognize_session = None
            return

        self._charset = self._load_charset()
        if not self._charset:
            logger.warning(
                "识别权重已加载但字符表缺失/为空: %s（第 1 行须为 CTC blank 占位）",
                self.charset_path,
            )
            self._recognize_session = None

    def _load_charset(self) -> list[str]:
        """读字符表。第 1 行是 CTC blank 占位（内容随意，约定不下标不对齐）。"""
        path = self.charset_path
        if not path.is_file():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.warning("字符表读取失败: %s", path, exc_info=True)
            return []
        # 逐行读、不做 strip：字符表里可能真有空格这一项
        chars = [line.rstrip("\r\n") for line in lines]
        while chars and not chars[-1]:
            chars.pop()
        return chars

    def close(self) -> None:
        self._detect_session = None
        self._recognize_session = None
        self._charset = []
        if self._recognizer_engine is not None:
            try:
                self._recognizer_engine.close()
            except Exception:  # pragma: no cover - 关闭失败不影响主流程
                logger.debug("回退识别引擎关闭失败", exc_info=True)
        self._recognizer_engine = None
        self._ready = False

    # -------------------------------------------------------------- 推理

    def predict(self, image: np.ndarray | str | Path) -> list[PlateResult]:
        """识别一张图中的所有车牌。"""
        self.ensure_ready()

        if isinstance(image, (str, Path)):
            img = imread(image)
        else:
            img = image
        if img is None:
            logger.warning("图像读取失败: %s", image)
            return []

        check_image(img, Limits())

        started = time.perf_counter()
        detections = self._detect(img)
        if not detections:
            logger.debug("未检测到车牌候选")
            return []

        results = self._recognize_all(img, detections)
        elapsed = (time.perf_counter() - started) * 1000
        for r in results:
            r.meta.setdefault("latency_ms", round(elapsed, 1))

        results.sort(key=lambda r: plate_score(r.text) if r.text else r.confidence, reverse=True)
        logger.debug(
            "检测识别引擎 %.1f ms 检出 %d 个候选 → %d 条结果",
            elapsed,
            len(detections),
            len(results),
        )
        return results

    # -------------------------------------------------------------- 检测

    def _detect(self, img: np.ndarray) -> list[Detection]:
        """跑检测网络并解码成 :class:`Detection` 列表。

        流程：letterbox → 前向 → 解析布局 → 解码 → 阈值 → NMS →
        逆 letterbox → 几何合理性过滤 → 角点排序。
        """
        blob, ratio, (pad_x, pad_y) = self._preprocess(img)
        outputs = self._infer(blob)
        pred, spec = self._select_output(outputs)
        self._spec = spec

        boxes, scores, class_ids, points = decode_predictions(pred, spec)
        if boxes.size == 0:
            return []

        keep = np.nonzero(scores >= self.conf_threshold)[0]
        if keep.size == 0:
            return []
        kept = nms(
            boxes[keep],
            scores[keep],
            self.iou_threshold,
            class_ids[keep] if spec.num_classes > 1 else None,
        )
        picked = keep[np.asarray(kept, dtype=np.int64)]

        h, w = img.shape[:2]
        detections: list[Detection] = []
        for i in picked:
            det = self._build_detection(
                boxes[i],
                points[i] if points is not None else None,
                float(scores[i]),
                int(class_ids[i]),
                ratio,
                (pad_x, pad_y),
                (w, h),
            )
            if det is not None:
                detections.append(det)

        logger.debug(
            "检测：候选 %d → 阈值后 %d → NMS 后 %d → 几何过滤后 %d（布局 %s）",
            len(scores),
            int(keep.size),
            len(picked),
            len(detections),
            spec,
        )
        return detections

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """BGR 图 → 归一化的 ``(1, 3, S, S)`` 浮点张量。"""
        padded, ratio, pad = letterbox(img, self.input_size)
        if self.rgb:
            padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(blob), ratio, pad

    def _infer(self, blob: np.ndarray) -> list[np.ndarray]:
        """一次前向，返回全部输出张量。"""
        if self._detect_session is None:
            raise EngineUnavailable("检测会话未加载，请先调用 load()")
        outputs = self._detect_session.run(None, {self._input_name: blob})
        return [np.asarray(o) for o in outputs]

    def _select_output(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, OutputSpec]:
        """在多个输出里挑出**能解析成检测布局**的那个。

        导出的 ONNX 常带辅助输出（如特征图、``416`` 之类），索引 0 不一定
        是最终预测。逐个试解析比硬编码索引 0 稳。
        """
        errors: list[str] = []
        for idx, out in enumerate(outputs):
            try:
                spec = resolve_output_spec(
                    out.shape,
                    num_classes=self.num_classes,
                    point_dim=self.point_dim,
                    has_objectness=self.has_objectness,
                    attr_order=self.attr_order,
                )
            except ValueError as exc:
                errors.append(f"[{idx}] {out.shape}: {exc}")
                continue
            return out, spec

        raise EngineUnavailable(
            "检测模型输出无法解析："
            + "；".join(errors)
            + "。请确认权重是本引擎期望的车牌检测模型，或用 num_classes / point_dim / "
            "has_objectness / attr_order 显式指明输出布局。"
        )

    def _build_detection(
        self,
        box: np.ndarray,
        points: np.ndarray | None,
        score: float,
        class_id: int,
        ratio: float,
        pad: tuple[int, int],
        image_size: tuple[int, int],
    ) -> Detection | None:
        """把 letterbox 坐标系的一条预测还原到原图，并做几何过滤。"""
        pad_x, pad_y = pad
        img_w, img_h = image_size

        xyxy = np.asarray(box, dtype=np.float32).copy()
        xyxy[[0, 2]] = (xyxy[[0, 2]] - pad_x) / ratio
        xyxy[[1, 3]] = (xyxy[[1, 3]] - pad_y) / ratio

        if points is not None:
            quad = np.asarray(points, dtype=np.float32).copy()
            quad[:, 0] = (quad[:, 0] - pad_x) / ratio
            quad[:, 1] = (quad[:, 1] - pad_y) / ratio
        else:
            # 纯检测权重没有角点分支：用外接框四角退化代替
            quad = np.array(
                [
                    [xyxy[0], xyxy[1]],
                    [xyxy[2], xyxy[1]],
                    [xyxy[2], xyxy[3]],
                    [xyxy[0], xyxy[3]],
                ],
                dtype=np.float32,
            )

        quad = _order_quad(quad)
        x1, y1 = float(quad[:, 0].min()), float(quad[:, 1].min())
        x2, y2 = float(quad[:, 0].max()), float(quad[:, 1].max())

        # 完全落在画面外 → 丢弃（不裁剪：裁剪后的框会误导后续评估）
        if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
            return None

        warped_w, warped_h = _warped_size(quad)
        if warped_h < _MIN_WARPED_HEIGHT:
            return None
        aspect = warped_w / max(1.0, warped_h)
        if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
            logger.debug("丢弃宽高比 %.2f 的候选（%.0f×%.0f）", aspect, warped_w, warped_h)
            return None

        layout = PlateLayout.DOUBLE if class_id in self.double_class_ids else PlateLayout.SINGLE
        return Detection(
            corners=quad,
            box=(x1, y1, x2, y2),
            layout=layout,
            score=float(score),
            class_id=class_id,
        )

    # -------------------------------------------------------------- 识别

    def _recognize_all(self, img: np.ndarray, detections: list[Detection]) -> list[PlateResult]:
        """对每个检测框做透视矫正后识别字符。

        三个后端按优先级选择（见 :meth:`_recognize_label`）：内置 CRNN →
        复用 ``deep`` 引擎识别管线 → 纯检测（``text`` 为空但保留框）。
        """
        # 一次性把「矫正图 + 框 + 版式 + 检测分」绑在一起，后面的分支都吃
        # 同一份数据；透视失败的候选在这里就被丢掉，不用每个分支各判一次。
        prepared: list[tuple[np.ndarray, tuple[int, int, int, int], PlateLayout, float]] = []
        for det in detections:
            roi = self._warp(img, det.corners)
            if roi is None or roi.size == 0:
                continue
            prepared.append((roi, _to_xywh(det), det.layout, det.score))
        if not prepared:
            return []

        if self._recognize_session is not None and self._charset:
            return [
                self._recognize_one_crnn(roi, bbox, layout, score)
                for roi, bbox, layout, score in prepared
            ]

        recognizer = self._get_recognizer()
        if recognizer is not None:
            regions = [(roi, PlateColor.UNKNOWN, bbox) for roi, bbox, _l, _s in prepared]
            results = recognizer.recognize_regions(
                regions, layout_hint=[layout for _r, _b, layout, _s in prepared]
            )
            # ``recognize_regions`` **会丢掉一个读数都没有的候选**（多尺度
            # 全轮无产出时它直接 ``continue``），所以返回列表通常比入参短。
            # 早先用 ``zip(results, prepared)`` 按位置配对 —— 只要中间丢一个，
            # 后面全部错位：A 的识别结果会挂上 B 的框和检测分。改为**按 bbox
            # 配对**（bbox 是我们自己传进去的，deep 原样写回，可精确匹配）。
            # 键用 ``object`` 而不是固定长度的元组类型：``PlateResult.bbox``
            # 声明成不定长元组，写成 ``set[tuple[int, int, int, int]]`` 会与
            # 它不兼容（mypy 会拦），而这里的键本来就只用来做相等匹配。
            by_bbox: dict[object, tuple[np.ndarray, PlateLayout, float]] = {
                bbox: (roi, layout, score) for roi, bbox, layout, score in prepared
            }
            matched: set[object] = set()
            merged: list[PlateResult] = []
            for result in results:
                hit = by_bbox.get(result.bbox) if result.bbox is not None else None
                if hit is not None:
                    matched.add(result.bbox)
                    result.meta.setdefault("det_score", round(float(hit[2]), 4))
                _retag(result, self.name)
                merged.append(result)

            # 检测到、但识别侧一个字都没读出来的候选**不能丢**：定位本身是
            # 有效产出（纯检测模式就是靠这个），静默丢弃会让用户以为检测
            # 也失败了。保留框、文本留空，并标注 ocr_failed 便于区分
            # 「没检到」和「检到了没认出」。
            for roi, bbox, layout, score in prepared:
                if bbox in matched:
                    continue
                merged.append(
                    PlateResult(
                        text="",
                        color=PlateColor.UNKNOWN,
                        layout=layout,
                        bbox=bbox,
                        roi=roi,
                        confidence=float(score),
                        engine=self.name,
                        meta={
                            "recognize_backend": "deep",
                            "ocr_failed": True,
                            "det_score": round(float(score), 4),
                        },
                    )
                )
            if len(merged) != len(results):
                logger.debug(
                    "识别侧丢弃 %d 个候选，已按纯检测口径补回（保框不保文本）",
                    len(merged) - len(results),
                )
            return merged

        # 纯检测模式：返回带框但无文本的结果。
        # 这不是失败而是明确的降级 —— 定位评测（如 tools/eval_ccpd.py）
        # 只需要框，静默返回空列表反而会让人以为模型坏了。
        logger.debug("无可用识别后端，返回纯检测结果（%d 个候选）", len(prepared))
        return [
            PlateResult(
                text="",
                color=PlateColor.UNKNOWN,
                layout=layout,
                bbox=bbox,
                roi=roi,
                confidence=float(score),
                engine=self.name,
                meta={"recognize_backend": "none", "det_score": round(float(score), 4)},
            )
            for roi, bbox, layout, score in prepared
        ]

    def _recognize_one_crnn(
        self, roi: np.ndarray, bbox: tuple[int, int, int, int], layout: PlateLayout, score: float
    ) -> PlateResult:
        """用自带的识别权重（CRNN + CTC）读一个候选区。"""
        blob = self._preprocess_rec(roi)
        raw = self._recognize_session.run(None, {self._rec_input_name: blob})[0]
        text, char_probs, rec_score = ctc_decode(raw, self._charset)
        text = _clean_plate_text(text)
        corrected, changes = correct(text) if text else ("", [])

        from .locate import detect_color

        color, _ = detect_color(roi, self._locate_config())
        confidence = (
            float(np.clip(0.6 * rec_score + 0.4 * plate_score(corrected), 0, 1))
            if corrected
            else float(rec_score)
        )
        return PlateResult(
            text=corrected,
            color=color,
            layout=layout,
            bbox=bbox,
            roi=roi,
            confidence=confidence,
            engine=self.name,
            char_confidences=char_probs,
            meta={
                "recognize_backend": "crnn",
                "det_score": round(float(score), 4),
                "raw_text": text,
                "corrections": changes,
            },
        )

    def _preprocess_rec(self, roi: np.ndarray) -> np.ndarray:
        """车牌区域 → CRNN 输入张量 ``(1, 3, 48, W)``。"""
        h, w = roi.shape[:2]
        ratio = REC_INPUT_HEIGHT / max(1, h)
        new_w = max(1, min(REC_MAX_WIDTH, _round_int(w * ratio)))
        resized = cv2.resize(roi, (new_w, REC_INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
        # 右侧补 0 到固定宽度：识别网络的输入宽度是静态的
        if new_w < REC_MAX_WIDTH:
            resized = cv2.copyMakeBorder(
                resized, 0, 0, 0, REC_MAX_WIDTH - new_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # 归一化到 [-1, 1]（PaddleOCR 系 CRNN 的常规口径）
        rgb = (rgb - 0.5) / 0.5
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])

    def _get_recognizer(self) -> Any:
        """惰性取得识别回退引擎（``deep``）。

        只在真的需要用的时候才加载 OCR 模型 —— 纯检测评测场景不该被迫
        多装一个 RapidOCR。加载失败**只记一次**日志并永久降级到纯检测，
        否则每张图都会重复打印同一条警告。
        """
        if self.recognize_backend == "none" or self._recognizer_failed:
            return None
        if self._recognizer_engine is not None:
            return self._recognizer_engine

        try:
            from .deep_engine import DeepEngine

            cfg = self.config if isinstance(self.config, EngineConfig) else None
            engine = DeepEngine(config=cfg)
            engine.ensure_ready()
            self._recognizer_engine = engine
            logger.debug("识别回退引擎（deep）已就绪")
        except Exception:
            self._recognizer_failed = True
            logger.warning(
                "无法启用识别回退引擎（deep），本引擎将以**纯检测模式**运行；"
                "如需字符识别请安装 rapidocr-onnxruntime 或提供 plate_rec.onnx",
                exc_info=True,
            )
            return None
        return self._recognizer_engine

    def _warp(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
        """按四角点做透视矫正，得到正向矩形车牌图。

        复用 ``locate._warp_plate``（内部先 ``_order_quad``）：那里的角点
        排序已经解决了「侧倾 >45° 被转成竖图」的老缺陷（v2.3.4），
        这里再实现一遍只会把那个 bug 带回来。
        """
        quad = _order_quad(corners)
        x1, y1 = float(quad[:, 0].min()), float(quad[:, 1].min())
        x2, y2 = float(quad[:, 0].max()), float(quad[:, 1].max())
        bbox = (int(x1), int(y1), max(1, _round_int(x2 - x1)), max(1, _round_int(y2 - y1)))

        warped_w, warped_h = _warped_size(quad)
        # 内存保护：角点退化时目标尺寸可能爆掉
        img_h, img_w = img.shape[:2]
        if warped_w > 4 * img_w or warped_h > 4 * img_h:
            return None

        try:
            roi = _warp_plate(img, quad, bbox)
        except cv2.error:
            logger.debug("透视变换失败", exc_info=True)
            return None
        if roi is None or roi.size == 0 or min(roi.shape[:2]) < 2:
            return None
        return roi

    def _locate_config(self) -> LocateConfig:
        """取定位/判色参数：优先用传入的 ``EngineConfig.locate``。"""
        if isinstance(self.config, EngineConfig):
            return self.config.locate
        return LocateConfig()


# ============================================================ 小工具


def _warped_size(quad: np.ndarray) -> tuple[float, float]:
    """与 ``_warp_plate`` 一致的矫正后目标尺寸（宽, 高），取上下边/左右边较大者。"""
    tl, tr, br, bl = quad
    width = max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)))
    height = max(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)))
    return width, height


def _to_xywh(det: Detection) -> tuple[int, int, int, int]:
    """检测框 → ``(x, y, w, h)``（与 ``PlateResult.bbox`` 同口径）。"""
    x1, y1, x2, y2 = det.box
    return (
        _round_int(x1),
        _round_int(y1),
        max(1, _round_int(x2 - x1)),
        max(1, _round_int(y2 - y1)),
    )


def _retag(result: PlateResult, engine_name: str) -> None:
    """把复用识别管线产生的结果改标成当前引擎。

    ``deep`` 的识别管线会把 ``engine`` 写成 ``"deep"``、把后端写进
    ``ocr_backend``；这里改回本引擎名，同时把"谁识别的"记进 ``recognizer``，
    否则排查问题时会看不出结果其实是 deep 读出来的。
    """
    result.meta["recognizer"] = result.meta.get("ocr_backend", "deep")
    result.engine = engine_name


def _clean_plate_text(text: str) -> str:
    """去掉 CTC 输出里的分隔符与空白并统一大写。"""
    remove = set("·•.。-_—– \t\n:：,，;；|/\\()（）[]【】<>《》'\"`~!！?？")
    return "".join(ch for ch in text if ch not in remove).upper()


__all__ = [
    "ATTR_ORDERS",
    "ATTR_ORDER_CLS_FIRST",
    "ATTR_ORDER_KPT_FIRST",
    "CHARSET_NAME",
    "CTC_BLANK_INDEX",
    "DEFAULT_MODEL_DIR",
    "DETECT_MODEL_NAME",
    "LETTERBOX_VALUE",
    "RECOGNIZE_MODEL_NAME",
    "Detection",
    "OutputSpec",
    "YoloEngine",
    "ctc_decode",
    "decode_predictions",
    "letterbox",
    "locate_model_dir",
    "nms",
    "resolve_output_spec",
]
