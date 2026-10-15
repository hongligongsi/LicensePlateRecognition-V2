"""用 ultralytics 训练车牌检测模型并导出成 ``plate_det.onnx``。

这是 ``tools/ccpd_to_yolo.py`` 的下一步：数据集准备好之后，训练一个
YOLO-pose 模型（bbox + 4 角点 + 单/双层两类），再导出 ONNX 给
:class:`lpr.engines.yolo_engine.YoloEngine` 使用。

## 依赖

需要 ``ultralytics`` + ``torch``：**故意不放进本项目的安装依赖里**。
引擎推理只要 ``onnxruntime``（几十 MB），而训练栈是 2 GB 量级 ——
把训练依赖塞进运行时安装，等于让只想跑识别的人也下载整套训练环境::

    pip install ultralytics

## 用法

::

    # 1) 先转数据集（见 tools/ccpd_to_yolo.py 的说明）
    python tools/ccpd_to_yolo.py --ccpd datasets/ccpd_raw --out datasets/ccpd_pose

    # 2) 训练 + 导出
    python tools/train_plate_yolo.py --data datasets/ccpd_pose/data.yaml --export-onnx

    # 3) 落到引擎认的路径，并确认布局能被解析
    cp runs/pose/plate/best.onnx models/yolo/plate_det.onnx
    python tools/fetch_yolo_weights.py --inspect models/yolo/plate_det.onnx

## 一个必须知道的坑

导出后**不要**急着设 ``attr_order``：Ultralytics 导出的 pose 模型是
``box + cls + kpts``（``cls_first``），正好是本引擎的默认值，无需设置。
反而是接了别人按 ``box + obj + kpts + cls`` 写的权重时才要改成
``kpt_first``。用 ``--inspect`` 确认一下最稳。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: 默认训练轮数。CCPD 30 万张按 640 输入训到收敛通常在 100~200 轮。
DEFAULT_EPOCHS = 120
#: 默认基础权重。n 版最快、s 版精度更好；车牌是单类别小目标，n/s 通常够用。
DEFAULT_BASE_WEIGHTS = "yolo11n-pose.pt"
#: 导出时的输入尺寸 —— **必须**与运行时 YoloEngine(input_size=...) 一致
DEFAULT_IMGSZ = 640


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "✗ 需要 ultralytics 才能训练：pip install ultralytics\n"
            "  （训练依赖刻意没进本项目的安装依赖；推理只需要 onnxruntime）",
            file=sys.stderr,
        )
        return None
    return YOLO


def check_dataset(data_yaml: Path) -> int:
    """训练前先自检数据集，把能提前发现的错误挡在训练之前。"""
    if not data_yaml.is_file():
        print(f"✗ 找不到 {data_yaml}", file=sys.stderr)
        return 2

    text = data_yaml.read_text(encoding="utf-8")
    problems = []
    if "kpt_shape" not in text:
        problems.append("缺 kpt_shape（pose 训练必需）")
    if "flip_idx" not in text:
        problems.append("缺 flip_idx —— 翻转增广会把关键点学乱，且不报错")
    if "names:" not in text:
        problems.append("缺 names")

    root = data_yaml.parent
    for split in ("train", "val"):
        d = root / "images" / split
        if not d.is_dir():
            problems.append(f"缺目录 {d.relative_to(root)}")
        elif not any(d.iterdir()):
            problems.append(f"{d.relative_to(root)} 是空的")
    label_dir = root / "labels" / "train"
    if label_dir.is_dir():
        sample = next((p for p in label_dir.glob("*.txt")), None)
        if sample is not None:
            first = sample.read_text(encoding="utf-8").splitlines()
            if first:
                fields = len(first[0].split())
                # cls + 4 框 + 4×2 关键点 = 13
                if fields != 13:
                    problems.append(f"标注字段数 {fields}，应为 13（1 类别 + 4 框 + 8 关键点）")
    if problems:
        print("✗ 数据集自检未通过：", file=sys.stderr)
        for p in problems:
            print(f"    · {p}", file=sys.stderr)
        return 2
    print("✓ 数据集自检通过")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="训练车牌 YOLO-pose 检测模型并导出 ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", required=True, help="ccpd_to_yolo.py 产出的 data.yaml")
    ap.add_argument("--weights", default=DEFAULT_BASE_WEIGHTS, help="基础权重或模型 yaml")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ, help="训练/导出输入尺寸（须与运行时一致）"
    )
    ap.add_argument("--batch", type=int, default=-1, help="批大小（-1 = 自动）")
    ap.add_argument("--device", default=None, help="如 0 / cpu；默认交给 ultralytics 自动选")
    ap.add_argument("--project", default="runs/pose", help="训练输出根目录")
    ap.add_argument("--name", default="plate", help="本次训练的子目录名")
    ap.add_argument("--export-onnx", action="store_true", help="训练完顺手导出 ONNX")
    ap.add_argument("--export-only", metavar="PT", help="跳过训练，直接把已有 .pt 导出成 ONNX")
    ap.add_argument("--skip-check", action="store_true", help="跳过数据集自检（不建议）")
    ap.add_argument("--opset", type=int, default=12, help="ONNX opset（默认 12，兼容性最好）")
    args = ap.parse_args()

    YOLO = _require_ultralytics()
    if YOLO is None:
        return 2

    data_yaml = Path(args.data).resolve()

    # ---- 仅导出模式
    if args.export_only:
        pt = Path(args.export_only)
        if not pt.is_file():
            print(f"✗ 找不到权重 {pt}", file=sys.stderr)
            return 2
        return _export(YOLO, pt, args)

    if not args.skip_check:
        code = check_dataset(data_yaml)
        if code:
            return code

    if not data_yaml.is_file():
        print(f"✗ 找不到 {data_yaml}", file=sys.stderr)
        return 2

    print(f"开始训练：{args.weights} → {data_yaml}")
    model = YOLO(args.weights)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    results = model.train(**train_kwargs)

    best = (
        Path(getattr(results, "save_dir", Path(args.project) / args.name)) / "weights" / "best.pt"
    )
    print(f"\n训练完成，最佳权重 {best}")
    if not best.is_file():
        best = Path(args.project) / args.name / "weights" / "best.pt"
        if not best.is_file():
            print("✗ 没找到 best.pt，跳过导出", file=sys.stderr)
            return 1

    if args.export_onnx:
        return _export(YOLO, best, args)

    print(
        "\n下一步导出："
        f"\n  python tools/train_plate_yolo.py --data {data_yaml} --export-only {best}"
    )
    return 0


def _export(YOLO, pt: Path, args: argparse.Namespace) -> int:
    """把 ``.pt`` 导出成 ONNX，并**顺手检查布局**（否则用户可能接到一个读不了的模型）。"""
    print(f"导出 ONNX（imgsz={args.imgsz}, opset={args.opset}）…")
    model = YOLO(str(pt))
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset)
    print(f"已导出 {onnx_path}")

    dest_dir = ROOT / "models" / "yolo"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "plate_det.onnx"
    shutil.copy2(onnx_path, dest)
    print(f"已复制到引擎默认路径：{dest}")

    # 复用 fetch 工具的检查逻辑：确认形状真的能被解析，并提示列顺序
    print()
    from fetch_yolo_weights import inspect_model

    return inspect_model(dest)


if __name__ == "__main__":
    raise SystemExit(main())
