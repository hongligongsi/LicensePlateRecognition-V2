#!/usr/bin/env python
"""基于真实图片数据自动调优定位/判定阈值。

## 为什么是"无监督"

理想的调参需要有标注的图片（每张图的真实车牌号与位置）。但当数据源只提供
图片、标注丢失时（例如 ModelScope 会把 CCPD 的文件名重写成 ``sample_N.jpg``，
而 CCDP 的标注恰恰编码在原始文件名里），监督式评估就无从谈起。

本脚本因此采用**无监督一致性指标**：不判断"认对没有"，而判断
"算法在自己身上的表现是否自洽、是否给出了足够有信息量的结论"。

## 四个指标

1. **定位命中率** ``locate_rate``
   有候选框的图片占比。定位不到 → 后续识别必然为空，是硬性损失。

2. **定位收敛度** ``focus``
   候选框数量接近 1 时最高。一张图上报出 5 个候选通常意味着形态学参数
   过于激进（噪声被连成块）；一个都没有则过于保守。

3. **判定确定率** ``decisive``
   候选区里"颜色 ≠ 未知"且"版式 ≠ 未知"的占比。参数调得好，算法应该
   对大部分车牌区域给出明确结论；大量 unknown 说明阈值不合适。

4. **几何合理度** ``geom`` —— **最关键的一项**
   车牌是**标准件**：单行 440×140mm（≈3.14:1）、双层 440×220mm（≈2:1）。
   一个候选框如果既不像单行牌又不像双层牌，几乎可以断定它是误检。
   本项统计"长宽比落在车牌合理区间内"的候选占比，直接惩罚把车身、
   保险杠、车灯误圈成车牌的情况。

综合得分 ``score = 0.35*locate_rate + 0.20*focus + 0.15*decisive + 0.30*geom``。

## 用法

    python tools/tune_thresholds.py datasets/ccpd_raw -n 200
    python tools/tune_thresholds.py datasets/ccpd_raw -n 200 --grid full
    python tools/tune_thresholds.py datasets/ccpd_raw -n 200 --apply

``--apply`` 会把最优参数写入 ``config.tuned.json``（不自动覆盖 config.json）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from lpr.config import EngineConfig, LocateConfig  # noqa: E402
from lpr.engines.deep_engine import measure_layout  # noqa: E402
from lpr.engines.locate import detect_color, locate_plates  # noqa: E402
from lpr.types import PlateColor  # noqa: E402
from lpr.utils.image import imread  # noqa: E402


# ------------------------------------------------------------------ 参数空间

#: 待搜索的参数网格。每一项是 ``(参数名, 候选值列表)``。
#: 取值区间参考真实车牌几何（单行 440x140mm ≈ 3.1:1，双层 440x220mm ≈ 2:1）。
GRID_QUICK = {
    "min_area_ratio": [0.0008, 0.0015, 0.0030],
    "min_aspect": [1.8, 2.0, 2.4],
    "color_margin": [0.02, 0.04, 0.08],
}

GRID_FULL = {
    "min_area_ratio": [0.0008, 0.0015, 0.0030, 0.0050],
    "min_aspect": [1.8, 2.0, 2.2, 2.4],
    "max_aspect": [4.5, 5.5, 6.5],
    "color_margin": [0.02, 0.04, 0.06, 0.08],
    "morph_kernel_ratio": [(0.004, 0.018), (0.005, 0.021), (0.006, 0.025)],
    "canny_low": [80, 100, 120],
}


@dataclass
class Metrics:
    """一轮评估的指标汇总。"""

    locate_rate: float = 0.0
    focus: float = 0.0
    decisive: float = 0.0
    geom: float = 0.0
    score: float = 0.0
    n_images: int = 0
    n_candidates: int = 0
    n_colored: int = 0
    n_layout_known: int = 0
    n_geom_ok: int = 0

    def as_dict(self) -> dict:
        return {
            "locate_rate": round(self.locate_rate, 4),
            "focus": round(self.focus, 4),
            "decisive": round(self.decisive, 4),
            "geom": round(self.geom, 4),
            "score": round(self.score, 4),
            "n_images": self.n_images,
            "n_candidates": self.n_candidates,
            "n_colored": self.n_colored,
            "n_layout_known": self.n_layout_known,
            "n_geom_ok": self.n_geom_ok,
        }


#: 车牌长宽比的合理区间（含少量透视余量）。
#: 单行 440x140mm ≈ 3.14:1，双层 440x220mm ≈ 2.0:1。
#: 实测含俯仰角/侧倾的实拍图，比例会落在 1.6~4.2 之间。
GEOM_MIN_ASPECT = 1.6
GEOM_MAX_ASPECT = 4.2


def _geom_ok(w: int, h: int) -> bool:
    """候选框的长宽比是否像一块真实车牌。"""
    if h <= 0 or w <= 0:
        return False
    a = w / h
    return GEOM_MIN_ASPECT <= a <= GEOM_MAX_ASPECT


# ------------------------------------------------------------------ 评估

def evaluate(images: list[np.ndarray], cfg: LocateConfig,
             engine_cfg: EngineConfig) -> Metrics:
    """在给定参数下评估一组图片，返回指标。"""
    hit = 0            # 至少有一个候选的图片数
    cand_counts: list[int] = []
    n_cand = 0
    n_colored = 0
    n_layout = 0
    n_geom = 0
    focus_sum = 0.0

    for img in images:
        try:
            cands, _ = locate_plates(img, cfg)
        except Exception:
            cand_counts.append(0)
            continue

        k = len(cands)
        cand_counts.append(k)
        n_cand += k
        if k > 0:
            hit += 1
            # 候选数量收敛度：1 个满分，之后递减（1→1.0, 2→0.7, 3→0.5, 5+→0.2）
            focus_sum += 1.0 / (1.0 + 0.45 * (k - 1))

        for c in cands:
            # 几何合理度：用定位阶段算好的 aspect（已含最小外接矩形）。
            # 退化时再退回按 bbox 现算。
            aspect = getattr(c, "aspect", 0.0) or 0.0
            if aspect <= 0:
                bx0, by0, bx1, by1 = c.bbox
                h_box = max(1, by1 - by0)
                aspect = (bx1 - bx0) / h_box
            if GEOM_MIN_ASPECT <= aspect <= GEOM_MAX_ASPECT:
                n_geom += 1

            roi = c.roi
            if roi is None or roi.size == 0:
                continue
            try:
                color, _ = detect_color(roi, cfg)
            except Exception:
                color = PlateColor.UNKNOWN
            try:
                layout = measure_layout(roi, engine_cfg)
            except Exception:
                layout = "unknown"
            if color not in (PlateColor.UNKNOWN,):
                n_colored += 1
            if layout != "unknown":
                n_layout += 1

    n = max(1, len(images))
    locate_rate = hit / n
    focus = focus_sum / n
    decisive = (
        (n_colored + n_layout) / (2.0 * n_cand) if n_cand > 0 else 0.0
    )
    geom = n_geom / n_cand if n_cand > 0 else 0.0
    score = (0.35 * locate_rate + 0.20 * focus
             + 0.15 * decisive + 0.30 * geom)

    return Metrics(locate_rate, focus, decisive, geom, score,
                   len(images), n_cand, n_colored, n_layout, n_geom)


# ------------------------------------------------------------------ 主流程

def load_images(directory: Path, limit: int) -> list[np.ndarray]:
    paths = sorted(directory.glob("*.jpg"))[:limit]
    out = []
    for p in paths:
        img = imread(p)
        if img is not None and img.size:
            out.append(img)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="定位/判定阈值自动调优")
    ap.add_argument("image_dir", help="图片目录（如 datasets/ccpd_raw）")
    ap.add_argument("-n", "--num", type=int, default=200, help="使用前 N 张图")
    ap.add_argument("--grid", choices=["quick", "full"], default="quick")
    ap.add_argument("--apply", action="store_true",
                    help="把最优参数写入 config.tuned.json")
    ap.add_argument("-o", "--out", default="out/tuning", help="报告输出目录")
    args = ap.parse_args()

    src = Path(args.image_dir)
    if not src.is_dir():
        print(f"目录不存在: {src}")
        return 1

    print(f"载入图片（前 {args.num} 张）...")
    images = load_images(src, args.num)
    if not images:
        print("没有可用图片")
        return 1
    print(f"  载入 {len(images)} 张，尺寸示例 {images[0].shape}")

    # 按固定步长交错划分 train/val，避免相邻帧高度相似导致乐观偏差
    if len(images) >= 40:
        val = images[::5]
        train = [im for i, im in enumerate(images) if i % 5 != 0]
    else:
        train, val = images, images
    print(f"  划分: 调参集 {len(train)} 张 / 验证集 {len(val)} 张")

    base = LocateConfig()
    engine_cfg = EngineConfig()
    grid = GRID_QUICK if args.grid == "quick" else GRID_FULL

    print("\n[基线|调参集] ", end="", flush=True)
    t0 = time.time()
    baseline = evaluate(train, base, engine_cfg)
    print(json.dumps(baseline.as_dict()), f"({time.time()-t0:.0f}s)")

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"\n搜索 {len(combos)} 组参数，共 {len(keys)} 个维度...")

    results = []
    t0 = time.time()
    for i, combo in enumerate(combos, 1):
        overrides = dict(zip(keys, combo))
        try:
            cfg = replace(base, **overrides)
        except (ValueError, TypeError):
            continue
        m = evaluate(train, cfg, engine_cfg)
        results.append((m.score, overrides, m))
        if i % 20 == 0 or i == len(combos):
            best = max(results, key=lambda r: r[0]) if results else None
            eta = (time.time() - t0) / i * (len(combos) - i)
            print(f"  {i}/{len(combos)}  当前最优 {best[0]:.4f}  剩余约 {eta:.0f}s",
                  flush=True)

    if not results:
        print("没有有效结果")
        return 1

    results.sort(key=lambda r: -r[0])
    best_score, best_params, best_m = results[0]

    # ---- 在**验证集**上复核，排除过拟合
    print("\n[验证] 用最优参数在独立验证集上复核...")
    val_base = evaluate(val, base, engine_cfg)
    val_best = evaluate(val, replace(base, **best_params), engine_cfg)

    print(f"\n{'='*72}")
    print(f"{'':10}{'调参集':>12}{'验证集':>12}")
    print(f"{'基线':10}{baseline.score:>12.4f}{val_base.score:>12.4f}")
    print(f"{'最优':10}{best_score:>12.4f}{val_best.score:>12.4f}")
    print(f"{'提升':10}{best_score-baseline.score:>+12.4f}{val_best.score-val_base.score:>+12.4f}")

    print("\n最优参数:")
    for k, v in best_params.items():
        print(f"  {k}: {getattr(base, k)}  ->  {v}")

    print("\nTop 10 组合（调参集）:")
    for s, p, m in results[:10]:
        ps = " ".join(f"{k}={v}" for k, v in p.items())
        print(f"  {s:.4f}  loc={m.locate_rate:.3f} foc={m.focus:.3f} "
              f"dec={m.decisive:.3f} geo={m.geom:.3f}  {ps}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "created": stamp,
        "image_dir": str(src),
        "n_images": len(images),
        "n_train": len(train),
        "n_val": len(val),
        "grid": args.grid,
        "n_combos": len(results),
        "train_baseline": baseline.as_dict(),
        "val_baseline": val_base.as_dict(),
        "best": {
            "score": best_score,
            "params": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in best_params.items()},
            "train_metrics": best_m.as_dict(),
            "val_metrics": val_best.as_dict(),
        },
        "top10": [
            {"score": s, "params": {k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in p.items()},
             "metrics": m.as_dict()}
            for s, p, m in results[:10]
        ],
    }
    rp = out_dir / f"tuning_{stamp}.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {rp}")

    if args.apply:
        from tune_io import apply_locate_params

        cfg_path = apply_locate_params(best_params, ROOT)
        print(f"已写入调优配置: {cfg_path}")
        print(f"  （原文件已备份为 {cfg_path.name}.bak）")
        print("  注意：config.json 的 engine.locate 会覆盖代码默认值，"
              "下次运行即生效。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
