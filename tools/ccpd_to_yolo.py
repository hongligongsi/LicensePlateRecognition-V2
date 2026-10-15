"""把 CCPD 转成 YOLO-pose 训练集（自训 ``plate_det.onnx`` 的第一步）。

## 产物

::

    <out>/
    ├── data.yaml             # ultralytics 的训练配置
    ├── images/train/*.jpg    # 或 symlink / hardlink
    ├── images/val/*.jpg
    ├── labels/train/*.txt    # YOLO-pose 标注
    └── labels/val/*.txt

标注格式（每行一个车牌）::

    cls  cx cy w h  x1 y1 x2 y2 x3 y3 x4 y4

全部按图像宽高归一化到 0~1。``cls`` 是 0=单行 / 1=双层。

## 两个容易静默毁掉训练的地方

**1. 四角点顺序。** CCPD 文件名里的四点是「右下 → 左下 → 左上 → 右上」，
而 YOLO-pose 要求的是「左上 → 右上 → 右下 → 左下」。本脚本按
``[2, 3, 0, 1]`` 重排。写错顺序**不会报错**，只会训出一个把车牌框
歪着套上去的模型 —— 训练 loss 还照常下降。

**2. 翻转增广的 ``flip_idx``。** 水平翻转后「左」和「右」互换，关键点必须
跟着重排。顺序为 [LT, RT, RB, LB] 时，翻转映射是 0↔1、3↔2，
即 ``flip_idx: [1, 0, 3, 2]``。不写这个字段，模型会把水平翻转过的样本
学成"角点乱序"，同样不报错。

## 用法

只看看能转多少张（不落盘）::

    python tools/ccpd_to_yolo.py --ccpd datasets/ccpd_raw --dry-run

正式转换（默认用软链，不占额外磁盘）::

    python tools/ccpd_to_yolo.py --ccpd datasets/ccpd_raw --out datasets/ccpd_pose --limit 20000
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ccpd_labels import CcpdAnnotation, load_dir

#: CCPD 四点顺序（右下/左下/左上/右上）→ 目标顺序（左上/右上/右下/左下）的置换。
#:
#: 即 ``target = [src[2], src[3], src[0], src[1]]``。
CCPD_TO_YOLO_CORNER_ORDER = (2, 3, 0, 1)

#: 类别名（与 YoloEngine.double_class_ids 默认值 (1,) 对应）
CLASS_NAMES = ("single", "double")

#: 水平翻转后关键点的置换：顺序 [LT, RT, RB, LB] 下 0↔1、3↔2
FLIP_IDX = (1, 0, 3, 2)

#: 判定「双层牌」的短边/长边比阈值。
#:
#: 单行牌 440×140mm → 0.318；双层牌 440×220mm → 0.50。
#: 取 0.42 落在两者中间，对透视误差留了余量。
DOUBLE_ASPECT_THRESHOLD = 0.42

#: 四点退化（近乎共线/重合）时判为脏标注的比例阈值：
#: 任意一边长度小于长边的该比例，认为标注不可用。
DEGENERATE_EDGE_RATIO = 0.15


@dataclass(frozen=True)
class PoseBox:
    """一条 YOLO-pose 标注。"""

    class_id: int
    #: 归一化的中心与宽高
    cx: float
    cy: float
    w: float
    h: float
    #: 4 个归一化角点，顺序 左上/右上/右下/左下
    points: tuple[tuple[float, float], ...]

    def to_line(self) -> str:
        coords = " ".join(f"{v:.6f}" for v in (self.cx, self.cy, self.w, self.h))
        pts = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        return f"{self.class_id} {coords} {pts}"


def _edge_lengths(corners: list[tuple[int, int]], order: tuple[int, ...]) -> tuple[float, float]:
    """按目标顺序取四点后，返回（平均宽, 平均高）。"""
    pts = [corners[i] for i in order]
    lt, rt, rb, lb = pts

    def dist(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    width = (dist(lt, rt) + dist(lb, rb)) / 2.0
    height = (dist(lt, lb) + dist(rt, rb)) / 2.0
    return width, height


def annotate(ann: CcpdAnnotation, width: int, height: int) -> PoseBox | None:
    """把一条 CCPD 标注转成 :class:`PoseBox`；不可用时返回 ``None``。

    ``width`` / ``height`` 是原图尺寸，用于归一化。
    """
    if width <= 0 or height <= 0 or len(ann.corners) != 4:
        return None

    span_x = [p[0] for p in ann.corners]
    span_y = [p[1] for p in ann.corners]
    x1, x2 = max(0.0, min(span_x)), min(float(width), max(span_x))
    y1, y2 = max(0.0, min(span_y)), min(float(height), max(span_y))
    if x2 - x1 < 4 or y2 - y1 < 2:
        return None

    edge_w, edge_h = _edge_lengths(ann.corners, CCPD_TO_YOLO_CORNER_ORDER)
    if edge_w <= 0 or edge_h <= 0:
        return None
    # 退化四边形（标注坏掉）会让边界框与关键点严重不自洽，宁可丢掉
    if edge_h < DEGENERATE_EDGE_RATIO * edge_w:
        return None

    class_id = 1 if edge_h / edge_w >= DOUBLE_ASPECT_THRESHOLD else 0

    points = tuple(
        (
            min(1.0, max(0.0, ann.corners[i][0] / width)),
            min(1.0, max(0.0, ann.corners[i][1] / height)),
        )
        for i in CCPD_TO_YOLO_CORNER_ORDER
    )
    return PoseBox(
        class_id=class_id,
        cx=((x1 + x2) / 2.0) / width,
        cy=((y1 + y2) / 2.0) / height,
        w=(x2 - x1) / width,
        h=(y2 - y1) / height,
        points=points,
    )


def is_val(name: str, ratio: float) -> bool:
    """按**文件名哈希**确定性切分验证集。

    不用 ``random``：同一次转换跑两遍必须给出同一份划分，否则"换台机器
    重训一次"的结果没法对比。用哈希则天然可复现，且与遍历顺序无关。
    """
    if ratio <= 0:
        return False
    if ratio >= 1:
        return True
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return bucket < ratio


def render_data_yaml(root: Path) -> str:
    """生成 ultralytics 的训练配置。"""
    lines = [
        "# 由 tools/ccpd_to_yolo.py 生成 —— 手改前先看脚本头部说明",
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    lines += [
        "",
        "# 关键点形状：4 个点、每点 (x, y)",
        "kpt_shape: [4, 2]",
        "",
        "# 水平翻转时关键点的重排（顺序为 左上/右上/右下/左下）；",
        "# 缺这一行，翻转增广会把角点学乱，而且**不会报错**。",
        f"flip_idx: {list(FLIP_IDX)}",
        "",
    ]
    return "\n".join(lines)


def _link(src: Path, dst: Path, mode: str) -> None:
    """把源图放到目标位置。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:  # 跨卷等情况退回复制
            shutil.copy2(src, dst)
    else:  # symlink
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            # Windows 未开开发者模式时建软链会失败：硬链 → 复制逐级退回
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CCPD → YOLO-pose 训练集", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ccpd", required=True, help="CCPD 原始图片目录（文件名内嵌标注）")
    ap.add_argument("--out", help="输出目录")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例（默认 0.1）")
    ap.add_argument("--limit", type=int, default=0, help="最多转多少张（0=全部）")
    ap.add_argument(
        "--mode",
        default="symlink",
        choices=["symlink", "hardlink", "copy", "list"],
        help="图片落盘方式；list 只写 txt 清单不建目录结构（默认 symlink）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写任何文件")
    args = ap.parse_args()

    src_dir = Path(args.ccpd)
    if not src_dir.is_dir():
        print(f"✗ 目录不存在: {src_dir}", file=sys.stderr)
        return 2

    import cv2

    annotations = load_dir(src_dir)
    if not annotations:
        print(
            f"✗ {src_dir} 里没有可解析的 CCPD 标注。\n"
            "  需要的是**原始 CCPD**（文件名形如 025-95_113-154&383_…-0_0_22_27…-37-15.jpg）；\n"
            "  ModelScope 镜像把文件名重写成了 sample_N.jpg，标注已丢失，无法用于训练。",
            file=sys.stderr,
        )
        return 2

    if args.limit:
        annotations = annotations[: args.limit]

    out_root = Path(args.out) if args.out else None
    if not args.dry_run and out_root is None:
        print("✗ 非 --dry-run 时必须指定 --out", file=sys.stderr)
        return 2

    written = {0: 0, 1: 0}
    skipped = 0
    splits = {"train": 0, "val": 0}
    train_list: list[str] = []
    val_list: list[str] = []
    out_root_resolved = out_root.resolve() if out_root else None

    for ann in annotations:
        img = cv2.imread(str(ann.path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        pose = annotate(ann, w, h)
        if pose is None:
            skipped += 1
            continue

        split = "val" if is_val(ann.path.name, args.val_ratio) else "train"
        splits[split] += 1
        written[pose.class_id] += 1

        if args.dry_run:
            continue

        assert out_root is not None and out_root_resolved is not None
        if args.mode == "list":
            # 仅清单模式：ultralytics 能读 txt 清单，但要求图片路径里含
            # images/ 且 label 能同构替换，因此这里把清单写到 out 下并说明限制
            entry = str(ann.path.resolve())
            (train_list if split == "train" else val_list).append(entry)
        else:
            _link(ann.path, out_root / "images" / split / ann.path.name, args.mode)
            label = out_root / "labels" / split / f"{ann.path.stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(pose.to_line() + "\n", encoding="utf-8")

    if not args.dry_run:
        assert out_root is not None
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "data.yaml").write_text(render_data_yaml(out_root.resolve()), encoding="utf-8")
        if args.mode == "list":
            (out_root / "train.txt").write_text("\n".join(train_list) + "\n", encoding="utf-8")
            (out_root / "val.txt").write_text("\n".join(val_list) + "\n", encoding="utf-8")

    total = sum(written.values())
    print(f"可转换 {total} 张（跳过 {skipped} 张：读图失败 / 标注退化 / 尺寸异常）")
    print(f"  单行牌 {written[0]}  双层牌 {written[1]}")
    print(f"  训练 {splits['train']}  验证 {splits['val']}")
    if args.dry_run:
        print("\n（--dry-run：未写任何文件）")
        return 0
    assert out_root is not None
    print(f"\n已写入 {out_root}（模式 {args.mode}）")
    if args.mode == "list":
        print("  注意：list 模式只写了 train.txt / val.txt，未建 images//labels/ 结构。")
    print(f"  配置 {out_root / 'data.yaml'}")
    print(
        "\n下一步：\n"
        f"  python tools/train_plate_yolo.py --data {out_root / 'data.yaml'} --export-onnx"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
