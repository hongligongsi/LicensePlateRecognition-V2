"""用带标注的检测数据集客观评测（并调优）车牌定位。

与 :mod:`tools.tune_thresholds` 的区别
--------------------------------------
``tune_thresholds`` 用的是**无监督**启发式打分（定位命中率 / 候选收敛度 /
判定确定率）—— 因为 CCPD 镜像的标注被抹掉了，没有真值可用。

本脚本则要求数据集是标准 **YOLO 检测格式**::

    <root>/
        images/xxx.jpg
        labels/xxx.txt     # 每行: cls cx cy w h  （归一化到 0~1）

有了真值框就能算**客观指标**：IoU 匹配后的精确率 / 召回率 / F1、平均 IoU、
中心点归一化误差。这比任何手工设计的启发式分数都可靠 —— 分数涨了就是真的涨了。

用法::

    # 单轮评测（基线）
    python tools/eval_locate.py datasets/hg/test

    # 网格搜索并输出最优参数
    python tools/eval_locate.py datasets/hg/test --grid quick --apply
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lpr.config import LocateConfig
from lpr.engines.locate import locate_plates

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """一张图 + 它的真值框（像素坐标 xyxy）。"""

    image: np.ndarray
    gt: np.ndarray  # shape (N, 4)
    name: str


def load_labels(path: Path, w: int, h: int) -> np.ndarray:
    """读 YOLO 标注，转成像素坐标 xyxy。"""
    boxes = []
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _, cx, cy, bw, bh = (float(v) for v in parts[:5])
        except ValueError:
            continue
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32)


def load_dataset(root: Path, limit: int | None = None) -> list[Sample]:
    """从 YOLO 目录结构读入样本。"""
    img_dir = root / "images" if (root / "images").is_dir() else root
    lab_dir = root / "labels" if (root / "labels").is_dir() else root
    samples: list[Sample] = []

    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    for p in files[:limit] if limit else files:
        # cv2.imread 不支持中文路径，用 fromfile 兜底
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            continue
        h, w = img.shape[:2]
        gt = load_labels(lab_dir / f"{p.stem}.txt", w, h)
        samples.append(Sample(img, gt, p.name))
    return samples


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """把定位输出的 ``(x, y, w, h)`` 转成 ``(x1, y1, x2, y2)``。

    注意：本项目 ``PlateCandidate.bbox`` / ``PlateResult.bbox`` 一律是
    ``(x, y, w, h)``（见 ``locate._crop_with_bbox`` 与 ``tracker._iou``）。
    评测侧统一转成 xyxy 再算 IoU，避免口径混用。
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两两 IoU。``a`` (M,4) x ``b`` (N,4) -> (M,N)，均为 xyxy。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0).astype(np.float32)


@dataclass
class Report:
    """一轮评测的汇总。

    指标分两类，各有用途：

    * **IoU 类**（``recall30/50/70``、``mean_iou``）—— 需要框贴合得准。
      注意本项目的定位框紧贴**字符带**（出于 OCR 需要），比人工标注的
      "整块车牌"略小，因此 IoU@0.5 偏严；``recall30`` 更能反映"抓到没抓到"。
    * **中心命中类**（``center_hit``）—— 只要预测框中心落在真值框内即算命中，
      衡量"定位方向对不对"，对框大小不敏感。
    """

    n_images: int = 0
    n_gt: int = 0
    n_pred: int = 0
    tp: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_iou: float = 0.0
    center_err: float = 0.0
    top1_iou: float = 0.0
    top1_hit: float = 0.0
    recall30: float = 0.0
    recall50: float = 0.0
    recall70: float = 0.0
    center_hit: float = 0.0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n_images": self.n_images,
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "tp": self.tp,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_iou": round(self.mean_iou, 4),
            "center_err": round(self.center_err, 4),
            "top1_iou": round(self.top1_iou, 4),
            "top1_hit": round(self.top1_hit, 4),
            "recall30": round(self.recall30, 4),
            "recall50": round(self.recall50, 4),
            "recall70": round(self.recall70, 4),
            "center_hit": round(self.center_hit, 4),
            "fitness": round(self.fitness, 4),
            "seconds": round(self.seconds, 1),
        }

    @property
    def fitness(self) -> float:
        """给网格搜索用的单一目标函数。

        ``recall30`` 是主体（有没有抓到），``precision`` 防"乱抓一堆"，
        ``-center_err`` 奖励定位更准。三者加权，避免优化出"召回了但框偏到离谱"的
        参数。注意网格搜索就靠这个值排序。
        """
        return 0.50 * self.recall30 + 0.25 * self.recall50 + 0.25 * max(0.0, 1.0 - self.center_err)


def evaluate(samples: list[Sample], cfg: LocateConfig, iou_thr: float = 0.5) -> Report:
    """跑一轮定位，按 IoU 贪心匹配算指标。"""
    t0 = time.time()
    n_gt = n_pred = tp = 0
    ious: list[float] = []
    cerrs: list[float] = []
    top1_ious: list[float] = []
    top1_hits = 0
    used_images = 0
    pos_images = 0
    hit30 = hit50 = hit70 = center_hits = 0

    # 网格搜索用固定阈值统计，不受 --iou 影响
    FIXED = (0.3, 0.5, 0.7)

    for s in samples:
        try:
            cands, _ = locate_plates(s.image, cfg)
        except Exception:
            cands = []
        pred = (
            xywh_to_xyxy(np.asarray([c.bbox for c in cands], dtype=np.float32))
            if cands
            else np.zeros((0, 4), np.float32)
        )

        used_images += 1
        n_gt += len(s.gt)
        n_pred += len(pred)

        if not len(s.gt):
            continue
        pos_images += 1

        if len(pred):
            m = iou_matrix(pred, s.gt)
            best = float(m.max())
            top1_ious.append(best)
            if best >= iou_thr:
                top1_hits += 1
            if best >= FIXED[0]:
                hit30 += 1
            if best >= FIXED[1]:
                hit50 += 1
            if best >= FIXED[2]:
                hit70 += 1

            # 中心命中：任一预测框的中心落在某个真值框内
            pcx = (pred[:, 0] + pred[:, 2]) / 2
            pcy = (pred[:, 1] + pred[:, 3]) / 2
            inside = (
                (pcx[:, None] >= s.gt[None, :, 0])
                & (pcx[:, None] <= s.gt[None, :, 2])
                & (pcy[:, None] >= s.gt[None, :, 1])
                & (pcy[:, None] <= s.gt[None, :, 3])
            )
            if bool(inside.any()):
                center_hits += 1

            # 贪心匹配（全局 P/R）
            for i, j in _greedy_match(m, iou_thr):
                tp += 1
                ious.append(float(m[i, j]))
                pc = (pred[i, :2] + pred[i, 2:]) / 2
                gc = (s.gt[j, :2] + s.gt[j, 2:]) / 2
                diag = float(np.hypot(*(s.gt[j, 2:] - s.gt[j, :2])))
                cerrs.append(float(np.hypot(*(pc - gc))) / max(diag, 1e-6))
        else:
            top1_ious.append(0.0)

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = max(1, pos_images)

    return Report(
        n_images=used_images,
        n_gt=n_gt,
        n_pred=n_pred,
        tp=tp,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=float(np.mean(ious)) if ious else 0.0,
        center_err=float(np.mean(cerrs)) if cerrs else 1.0,
        top1_iou=float(np.mean(top1_ious)) if top1_ious else 0.0,
        top1_hit=top1_hits / denom,
        recall30=hit30 / denom,
        recall50=hit50 / denom,
        recall70=hit70 / denom,
        center_hit=center_hits / denom,
        seconds=time.time() - t0,
    )


def _greedy_match(m: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """按 IoU 从大到小贪心配对，返回 (pred_idx, gt_idx)。"""
    pairs: list[tuple[int, int]] = []
    if m.size == 0:
        return pairs
    order = np.argsort(-m, axis=None)
    used_p: set[int] = set()
    used_g: set[int] = set()
    for flat in order:
        i, j = divmod(int(flat), m.shape[1])
        if m[i, j] < thr or i in used_p or j in used_g:
            continue
        pairs.append((i, j))
        used_p.add(i)
        used_g.add(j)
    return pairs


# --------------------------------------------------------------------------- #
# 网格
# --------------------------------------------------------------------------- #
#: 搜索网格。范围必须**包住默认值**，否则"最优"只是网格边缘的伪最优。
#:
#: 默认值：min_area_ratio=0.0015，morph_kernel_ratio=(0.005, 0.021)，
#: open_kernel_ratio=0.02，min_aspect=2.0，max_aspect=5.5，box_padding=5(px)。
#:
#: 量纲提醒：``morph_kernel_ratio`` 与 ``open_kernel_ratio`` 都是**相对图宽/图高的
#: 比例**（不是千分比）。此前网格给到 (0.025, 0.60)，等于用图宽 60% 的核做形态学
#: 运算 —— 那会把整幅图糊成一团，选出来的"最优"是退化解，不能用。
GRID_QUICK = {
    "min_area_ratio": (0.0005, 0.0010, 0.0015, 0.0025),
    "morph_kernel_ratio": ((0.003, 0.012), (0.005, 0.021), (0.008, 0.030)),
}

GRID_FULL = {
    "min_area_ratio": (0.0005, 0.0008, 0.0015, 0.0022, 0.0035),
    "morph_kernel_ratio": (
        (0.003, 0.010),
        (0.004, 0.015),
        (0.005, 0.021),
        (0.007, 0.028),
        (0.010, 0.038),
    ),
    "open_kernel_ratio": (0.008, 0.014, 0.020, 0.028),
    "min_aspect": (1.8, 2.0, 2.2),
    "max_aspect": (4.5, 5.5, 6.5),
    "color_margin": (0.02, 0.04, 0.07),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="YOLO 数据集根目录（含 images/ 与 labels/）")
    ap.add_argument("-n", "--num", type=int, default=None, help="只用前 N 张")
    ap.add_argument("--iou", type=float, default=0.5, help="匹配 IoU 阈值")
    ap.add_argument("--grid", choices=["none", "quick", "full"], default="none")
    ap.add_argument("--apply", action="store_true", help="把最优参数写入 config.tuned.json")
    ap.add_argument("--out", default="reports", help="报告输出目录（相对项目根）")
    ap.add_argument(
        "--save-baseline",
        metavar="PATH",
        help="把本次基线指标存成回归基线文件（之后可用 --baseline 对照）",
    )
    ap.add_argument(
        "--baseline",
        metavar="PATH",
        help=(
            "对照已存的基线指标：fitness 掉超过 --tolerance 就以非零码退出。"
            "注意必须在**同一个数据集**上跑才可比。CI 无法内嵌带标注数据集，"
            "因此这是本地/自建 CI 用的能力"
        ),
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="允许的 fitness 跌幅（默认 0.01），超过即判为回归",
    )
    args = ap.parse_args()

    src = Path(args.dataset)
    if not src.is_absolute():
        src = ROOT / src
    if not src.is_dir():
        print(f"目录不存在: {src}")
        return 1

    print(f"加载 {src} ...")
    samples = load_dataset(src, args.num)
    if not samples:
        print("没有可用样本（检查 images/ 与 labels/ 结构）")
        return 1
    n_pos = sum(1 for s in samples if len(s.gt))
    print(f"  {len(samples)} 张（其中 {n_pos} 张含真值框），示例尺寸 {samples[0].image.shape}")

    base = LocateConfig()
    print("\n[基线] ", end="", flush=True)
    baseline = evaluate(samples, base, args.iou)
    print(json.dumps(baseline.as_dict()))

    best_params: dict = {}
    if args.grid != "none":
        grid = GRID_QUICK if args.grid == "quick" else GRID_FULL
        keys = list(grid)
        combos = list(itertools.product(*(grid[k] for k in keys)))
        print(f"\n搜索 {len(combos)} 组参数（{len(keys)} 维度）...")
        results = []
        t0 = time.time()
        for i, combo in enumerate(combos, 1):
            ov = dict(zip(keys, combo, strict=False))
            try:
                cfg = replace(base, **ov)
            except (ValueError, TypeError):
                continue
            m = evaluate(samples, cfg, args.iou)
            results.append((m.fitness, ov, m))
            if i % 10 == 0 or i == len(combos):
                b = max(results, key=lambda r: r[0])
                eta = (time.time() - t0) / i * (len(combos) - i)
                print(
                    f"  {i}/{len(combos)}  当前最优 fitness {b[0]:.4f}  剩 {eta:.0f}s", flush=True
                )
        if results:
            results.sort(key=lambda r: -r[0])
            best_fit, best_params, best_m = results[0]
            print(f"\n{'=' * 74}")
            print(
                f"基线 fitness {baseline.fitness:.4f}  ->  最优 {best_fit:.4f}  "
                f"({best_fit - baseline.fitness:+.4f})"
            )
            print(f"基线 {baseline.as_dict()}")
            print(f"最优 {best_m.as_dict()}")
            print("\n最优参数:")
            for k, v in best_params.items():
                print(f"  {k}: {getattr(base, k)}  ->  {v}")
            print("\nTop 8:")
            for fit, p, m in results[:8]:
                ps = " ".join(f"{k}={v}" for k, v in p.items())
                print(
                    f"  fit={fit:.4f} R@30={m.recall30:.3f} R@50={m.recall50:.3f} "
                    f"ctr={m.center_hit:.3f} preds={m.n_pred} | {ps}"
                )

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "created": stamp,
        "dataset": str(src),
        "n_samples": len(samples),
        "n_positive": n_pos,
        "iou_threshold": args.iou,
        "baseline": baseline.as_dict(),
        "best_params": best_params,
    }
    rp = out_dir / f"locate_eval_{stamp}.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {rp}")

    if args.apply and best_params:
        from tune_io import apply_locate_params

        cfg_path = apply_locate_params(best_params, ROOT)
        print(f"已写入调优配置: {cfg_path}")
        print(f"  （原文件已备份为 {cfg_path.name}.bak）")
        print("  注意：config.json 的 engine.locate 会覆盖代码默认值，下次运行即生效。")

    if args.save_baseline:
        bp = Path(args.save_baseline)
        if not bp.is_absolute():
            bp = ROOT / bp
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(
            json.dumps(
                {
                    "_note": (
                        "定位回归基线：用 --baseline 对照时，fitness 跌幅超过 "
                        "--tolerance 即判为回归。只在**同一数据集**上可比。"
                    ),
                    "dataset": str(src),
                    "n_samples": len(samples),
                    "iou_threshold": args.iou,
                    "metrics": baseline.as_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"回归基线已写入: {bp}")
        print(
            f"  之后可用: python tools/eval_locate.py {args.dataset} "
            f"--baseline {args.save_baseline}"
        )

    if args.baseline:
        return _check_against_baseline(args, baseline, len(samples))

    return 0


def _check_against_baseline(args, current: Report, n_samples: int) -> int:
    """把当前指标与基线对照，跌幅超阈值则返回非零码（可用于 CI 卡口）。"""
    bp = Path(args.baseline)
    if not bp.is_absolute():
        bp = ROOT / bp
    if not bp.exists():
        print(f"基线文件不存在: {bp}", file=sys.stderr)
        return 1

    saved = json.loads(bp.read_text(encoding="utf-8"))
    old = saved.get("metrics", {})
    print(f"\n{'=' * 74}\n对照基线: {bp}")
    if saved.get("n_samples") != n_samples:
        print(
            f"  ⚠ 样本数不一致（基线 {saved.get('n_samples')} vs 现在 {n_samples}），"
            "指标不可比，请在同数据集同切分上重跑",
            file=sys.stderr,
        )
        return 2

    old_fit = float(old.get("fitness", 0.0))
    new_fit = current.fitness
    delta = new_fit - old_fit
    print(f"  fitness: {old_fit:.4f} -> {new_fit:.4f}  ({delta:+.4f})")
    for key in ("recall30", "recall50", "mean_iou", "precision", "center_hit"):
        if key in old:
            print(f"  {key:<10}: {float(old[key]):.4f} -> {getattr(current, key):.4f}")

    if delta < -abs(args.tolerance):
        print(
            f"\n✗ 回归！fitness 跌幅 {-delta:.4f} 超过容忍 {abs(args.tolerance):.4f}",
            file=sys.stderr,
        )
        return 1
    print(f"\n✓ 未发现回归（容忍 {-abs(args.tolerance):+.4f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
