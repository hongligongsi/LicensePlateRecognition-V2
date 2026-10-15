"""生成**桩 ONNX 模型**，用于在拿到真权重之前自检整条推理链路。

## 为什么需要它

``yolo`` 引擎的权重不随仓库分发（体积 + 许可，见 ``models/yolo/README.md``）。
于是有个尴尬的空窗期：装完 extras、``lpr engines`` 显示"可用"之前，
用户没法确认「引擎代码 + onnxruntime + 你自己的后处理理解」这套组合是否
真的通。

本工具产出一个**输出为常量**的极小 ONNX 模型，属性列按你指定的布局摆好。
它当然不会真的检测车牌 —— 它验证的是**别的东西**：

* ONNX 会话能否建立、输入名能否取到、blob 形状/类型是否被接受；
* ``resolve_output_spec`` 对你这种布局的解析结果；
* letterbox → 解码 → 阈值 → NMS → 逆 letterbox → 角点矫正 这一整条
  坐标变换是否自洽（把指定的框还原回原图，应当落在预期位置）；
* ``lpr engines`` / CLI / 界面那条路是否真的把这个引擎跑通。

一句话：**它测的是管道，不是模型。**

## 用法

``--selftest`` 会顺手把生成的模型喂给真正的 :class:`YoloEngine`，把解码
出来的框打印出来 —— 这是最快的"我这套装好了没"检查::

    python tools/make_stub_onnx.py --layout kpt-first --out models/yolo/plate_det.onnx --selftest

仅用于自检/测试；**不要把它当成可用权重**，它的输出与输入图无关。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

#: 各布局的元信息：``(channels_first, has_objectness, num_points, attr 数, 说明)``
#:
#: ``attr 数``按 ``num_classes`` 写成 lambda，因为属性数随类别数变。
LAYOUTS: dict[str, dict] = {
    "v8-pose": {
        "channels_first": True,
        "has_objectness": False,
        "num_points": 4,
        "attr": lambda nc: 4 + nc + 8,
        "order": "cls_first",
        "note": "Ultralytics YOLOv8-pose：box + cls + kpts，输出 (1, attr, N)",
    },
    "v5-pose": {
        "channels_first": False,
        "has_objectness": True,
        "num_points": 4,
        "attr": lambda nc: 4 + 1 + nc + 8,
        "order": "cls_first",
        "note": "Ultralytics YOLOv5-pose：box + obj + cls + kpts，输出 (1, N, attr)",
    },
    "kpt-first": {
        "channels_first": False,
        "has_objectness": True,
        "num_points": 4,
        "attr": lambda nc: 4 + 1 + nc + 8,
        "order": "kpt_first",
        "note": "we0091234 系 plate_detect.onnx：box + obj + kpts + cls，输出 (1, N, attr)",
    },
    "v8-det": {
        "channels_first": True,
        "has_objectness": False,
        "num_points": 0,
        "attr": lambda nc: 4 + nc,
        "order": "cls_first",
        "note": "只有检测头的 v8：box + cls，无角点（角点退化为外接框）",
    },
    "v5-det": {
        "channels_first": False,
        "has_objectness": True,
        "num_points": 0,
        "attr": lambda nc: 4 + 1 + nc,
        "order": "cls_first",
        "note": "只有检测头的 v5：box + obj + cls，无角点",
    },
}


def _row(
    layout: str,
    cx: float,
    cy: float,
    w: float,
    h: float,
    obj: float,
    cls_scores: list[float],
    corners: list[float],
) -> list[float]:
    """按布局把一条预测拼成属性向量。"""
    if layout == "v8-pose":
        return [cx, cy, w, h, *cls_scores, *corners]
    if layout == "v5-pose":
        return [cx, cy, w, h, obj, *cls_scores, *corners]
    if layout == "kpt-first":
        return [cx, cy, w, h, obj, *corners, *cls_scores]
    if layout == "v8-det":
        return [cx, cy, w, h, *cls_scores]
    if layout == "v5-det":
        return [cx, cy, w, h, obj, *cls_scores]
    raise ValueError(f"未知布局 {layout!r}")


def build_output(
    layout: str,
    *,
    num_classes: int = 2,
    anchors: int = 8,
    box: tuple[float, float, float, float] = (320.0, 320.0, 200.0, 64.0),
    double: bool = False,
    duplicate: bool = False,
) -> np.ndarray:
    """构造模型的常量输出张量。

    Args:
        box: ``(cx, cy, w, h)``，**letterbox 之后的输入图坐标系**（如 640×640）。
        double: 命中的类别下标是否为 1（本引擎默认把类 1 当双层牌）。
        duplicate: 额外放一条几乎重合的框，用来验证 NMS 真的在生效。

    Returns:
        形状与 ``layout`` 匹配的 float32 张量，前 2 条是有效目标
        （``duplicate`` 时为 3 条），其余全零（分数为 0，会被阈值滤掉）。
    """
    meta = LAYOUTS[layout]
    attr = int(meta["attr"](num_classes))
    cx, cy, w, h = box
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    cls_scores = [0.05] * num_classes
    cls_scores[1 if double else 0] = 0.95

    rows = [
        _row(
            layout,
            cx,
            cy,
            w,
            h,
            0.9,
            cls_scores,
            [x1, y1, x2, y1, x2, y2, x1, y2] if meta["num_points"] else [],
        )
    ]
    if duplicate:
        # 与第一条 IoU ≈ 0.87，明显超过默认 0.45 阈值 → 必须被 NMS 压掉一条。
        # 分数**同时**压低 obj 与类别分：无 objectness 的布局（v8 系）只用
        # max(cls) 当置信度，只压 obj 的话两条分数会相等，NMS 保留哪条就
        # 变成未定义行为（argsort 对相等值不保证稳定）—— 测试会间歇性失败。
        dup_cls = list(cls_scores)
        dup_cls[1 if double else 0] = 0.6
        dx = w * 0.05
        rows.append(
            _row(
                layout,
                cx + dx,
                cy,
                w,
                h,
                0.7,
                dup_cls,
                [x1 + dx, y1, x2 + dx, y1, x2 + dx, y2, x1 + dx, y2] if meta["num_points"] else [],
            )
        )

    while len(rows) < anchors:
        rows.append(_row(layout, 0.0, 0.0, 0.0, 0.0, 0.0, [0.0] * num_classes, [0.0] * 8))

    arr = np.asarray(rows, dtype=np.float32)
    # 属性数必须与布局声明一致 —— 写错的话桩模型就成了"另一个"模型，
    # 端到端测试会以假乱真地通过（或失败得莫名其妙）。
    assert arr.shape[1] == attr, f"{layout}: 期望 {attr} 个属性，实际 {arr.shape[1]}"
    if meta["channels_first"]:
        arr = arr.T  # (N, attr) -> (attr, N)
        return arr[None, ...]
    return arr[None, ...]


def build_model(output: np.ndarray, *, input_size: int, input_name: str = "images"):
    """把常量输出包成一个 ONNX 模型（输入声明为 ``(1, 3, S, S)``）。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inp = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [1, 3, input_size, input_size]
    )
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(output.shape))
    const = numpy_helper.from_array(np.asarray(output, np.float32), name="stub_output")
    node = helper.make_node("Constant", inputs=[], outputs=["output0"], value=const)
    graph = helper.make_graph([node], "stub_plate_det", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    # onnxruntime 各版本支持的 ir_version 上限不同，压到 8 兼容性最好
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def selftest(model_path: Path, layout: str, num_classes: int, input_size: int) -> int:
    """把生成的模型喂给真正的 YoloEngine，打印解码结果。"""
    from lpr.engines.yolo_engine import YoloEngine

    meta = LAYOUTS[layout]
    if model_path.name != "plate_det.onnx":
        # 引擎按固定文件名找权重；自检必须走同一条路，否则测的就不是真实路径
        print(
            f"自检要求 --out 的文件名为 plate_det.onnx（当前 {model_path.name}）；"
            f"引擎只会加载 <models_dir>/plate_det.onnx",
            file=sys.stderr,
        )
        return 2

    engine = YoloEngine(
        models_dir=model_path.parent,
        input_size=input_size,
        num_classes=num_classes,
        attr_order=str(meta["order"]),
        recognize_backend="none",  # 自检只关心检测，别去加载 OCR
    )

    # 宽高比 != 1 的图能验证 letterbox 逆变换（缩放 + 上下/左右补边）是否自洽
    img = np.full((480, 800, 3), 128, np.uint8)
    engine.load()
    detections = engine._detect(img)
    print(f"\n--- 自检（输入 {img.shape[1]}×{img.shape[0]}，布局 {layout}）---")
    print(f"布局解析: {engine._spec}")
    if not detections:
        print("✗ 未解出任何检测框 —— 布局或坐标变换有问题")
        return 1
    for d in detections:
        print(
            f"  框={tuple(round(v, 1) for v in d.box)} "
            f"版式={d.layout.value} 类别={d.class_id} 分数={d.score:.3f}"
        )
        print(f"    角点={[[round(float(v), 1) for v in p] for p in d.corners]}")
    print(f"✓ 解出 {len(detections)} 个候选（重复框已被 NMS 合并）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="生成桩 ONNX 检测模型（仅用于自检/测试，不是可用权重）"
    )
    ap.add_argument(
        "--layout",
        default="v8-pose",
        choices=sorted(LAYOUTS),
        help="属性列布局；用哪种取决于你打算接的权重，默认 v8-pose",
    )
    ap.add_argument("--out", required=True, help="输出 .onnx 路径")
    ap.add_argument("--input-size", type=int, default=640, help="模型输入边长（默认 640）")
    ap.add_argument("--num-classes", type=int, default=2, help="类别数（默认 2）")
    ap.add_argument("--anchors", type=int, default=8, help="锚点槽位数（默认 8）")
    ap.add_argument(
        "--box",
        default="320,320,200,64",
        help="常量框 cx,cy,w,h（输入图坐标系；默认居中 200×64）",
    )
    ap.add_argument("--double", action="store_true", help="把命中类别设为 1（双层牌）")
    ap.add_argument("--duplicate", action="store_true", help="额外放一条重合框以验证 NMS")
    ap.add_argument("--selftest", action="store_true", help="生成后立即用 YoloEngine 解码验证")
    args = ap.parse_args()

    try:
        import onnx  # noqa: F401
    except ImportError:
        print('需要 onnx 才能生成模型：pip install -e ".[dev]"', file=sys.stderr)
        return 2

    try:
        box = tuple(float(v) for v in args.box.split(","))
    except ValueError:
        print(f"--box 需要 4 个逗号分隔的数，收到 {args.box!r}", file=sys.stderr)
        return 2
    if len(box) != 4:
        print(f"--box 需要 4 个值，收到 {len(box)} 个", file=sys.stderr)
        return 2

    output = build_output(
        args.layout,
        num_classes=args.num_classes,
        anchors=args.anchors,
        box=box,  # type: ignore[arg-type]
        double=args.double,
        duplicate=args.duplicate,
    )
    model = build_model(output, input_size=args.input_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(model.SerializeToString())
    print(f"已写出 {out_path}（布局 {args.layout}，输出形状 {tuple(output.shape)}）")
    print(f"  {LAYOUTS[args.layout]['note']}")
    print("  提醒：这是桩模型，输出与输入图无关，仅供自检。")

    if args.selftest:
        return selftest(out_path, args.layout, args.num_classes, args.input_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
