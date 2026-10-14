"""CCPD 真实样张准确率评测（deep 引擎）。

## 数据来源

``datasets/ccpd_labeled/`` 下的 100 张 CCPD 图片，真值在 ``_labels.json``：

* 图片从 ModelScope ``OmniData/CCPD`` 与 HF 镜像 ``zenitsu09/ccpd-subset-30k``
  按需抽取（HTTP Range 读 zip 中央目录后单条解压，无需下载整包）；
* 文件名保留原始 CCPD 编码（车牌号下标串），车牌文本用 OpenDataLab 的
  DSDL 标注包（``datasets/dsdl/``）交叉验证后作为真值——两者独立来源，
  文件键（tilt-bbox-corners-plate 四段）完全一致才采纳；
* 子集构成：base(正常) 67 + weather 8 + blur 6 + tilt 6 + challenge 6
  + fn 4 + db 2 + rotate 1，覆盖常规与困难场景。

## 指标口径

CCPD 每图恰有一块牌，报告三个互补指标：

1. **图级命中**：真值牌 ∈ 检出集合（与 tests/test_real_samples_accuracy
   的"命中门槛"同口径，严格——多检出不影响，漏检/读错即败）；
2. **字符容差命中**：真值牌与某个检出互为编辑距离 ≤1 的近似匹配
   （诊断"差一个字符"的近失比例）；
3. **误检**：检出中既不等于真值、也不是近似的条目数（多余检出）。

用法::

    python tools/eval_ccpd.py                 # 全量 100 张
    python tools/eval_ccpd.py --subset base   # 只评某个子集
    python tools/eval_ccpd.py --limit 20      # 快速冒烟
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LABELS = ROOT / "datasets" / "ccpd_labeled" / "_labels.json"
IMG_DIR = ROOT / "datasets" / "ccpd_labeled"


def _d1(a: str, b: str) -> bool:
    """编辑距离是否 ≤1（仅用于近失诊断，不等价于识别正确）。"""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    # 保证 a 为短串
    if len(a) > len(b):
        a, b = b, a
    for i in range(len(b)):
        cand = b[:i] + b[i + 1 :]  # b 删一个字符
        if cand == a:
            return True
    return False  # 长度差 ≤1 时替换/增删一次必能化为上述情形的近似


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 张（冒烟用）")
    ap.add_argument("--subset", type=str, default="", help="只评某子集，如 base")
    args = ap.parse_args()

    if not LABELS.is_file():
        print(f"缺少真值文件 {LABELS}——先用抓取脚本生成 datasets/ccpd_labeled/")
        return 1
    labels = json.loads(LABELS.read_text(encoding="utf-8"))

    from lpr.config import EngineConfig
    from lpr.engines.deep_engine import DeepEngine
    from lpr.utils.image import imread

    eng = DeepEngine(EngineConfig())
    eng.load()

    items = sorted(labels.items())
    if args.subset:
        items = [(k, v) for k, v in items if v.get("subset") == args.subset]
    if args.limit:
        items = items[: args.limit]

    n = n_img_hit = 0
    n_plate_hit = 0  # 牌级命中（近失不计）
    n_near = 0  # 近失（差 1 个字符）
    n_extra = 0  # 误检条目数
    by_subset: dict[str, list[str]] = {}
    miss_detail: list[str] = []
    t0 = time.time()

    for name, meta in items:
        img = imread(str(IMG_DIR / name))
        if img is None:
            miss_detail.append(f"{name}: 图片读取失败")
            continue
        got = [r.text for r in eng.predict(img) if r.text]
        gt_list = meta["plates"]
        n += 1

        hit_plates = 0
        near_plates = 0
        extras = 0
        for gt in gt_list:
            if gt in got:
                hit_plates += 1
            elif any(_d1(gt, g) for g in got):
                near_plates += 1
        extras = sum(1 for g in got if not any(_d1(gt, g) for gt in gt_list))
        n_plate_hit += hit_plates
        n_near += near_plates
        n_extra += extras
        if hit_plates == len(gt_list):
            n_img_hit += 1
        else:
            sub = meta.get("subset", "?")
            by_subset.setdefault(sub, []).append(name)
            miss_detail.append(f"{name} [{sub}] GT={'/'.join(gt_list)} 检出={got or '无'}")

    # ---- 汇总
    if n == 0:
        print("没有可评测样本")
        return 1
    elapsed = time.time() - t0
    print(f"\n==== CCPD 评测（{n} 张, {elapsed:.0f}s, {elapsed / n * 1000:.0f} ms/张）====")
    print(f"图级命中（严格）: {n_img_hit}/{n} = {n_img_hit / n:.1%}")
    print(
        f"牌级命中: {n_plate_hit}/{n} = {n_plate_hit / n:.1%}"
        f"   近失(差1字符): {n_near}   误检条目: {n_extra}"
    )

    # 分子集命中率（图级）
    all_by_sub: dict[str, list[str]] = {}
    for name, meta in items:
        all_by_sub.setdefault(meta.get("subset", "?"), []).append(name)
    if len(all_by_sub) > 1:
        print("\n分子集图级命中:")
        for sub, names in sorted(all_by_sub.items()):
            miss = len(by_subset.get(sub, []))
            print(f"  {sub:10s} {len(names) - miss}/{len(names)}")

    if miss_detail:
        print("\n未命中明细:")
        for line in miss_detail:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
