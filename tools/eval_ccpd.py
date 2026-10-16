"""CCPD 真实样张的**多引擎**准确率评测。

## 数据来源

``datasets/ccpd_labeled/`` 下的 100 张 CCPD 图片，真值在 ``_labels.json``：

* 图片从 ModelScope ``OmniData/CCPD`` 与 HF 镜像 ``zenitsu09/ccpd-subset-30k``
  按需抽取（HTTP Range 读 zip 中央目录后单条解压，无需下载整包）；
* 车牌文本用 OpenDataLab 的 DSDL 标注包（``datasets/dsdl/``）交叉验证后作为真值；
* **定位真值来自文件名里的 bbox 段**（``x1,y1_x2,y2``）。该字段与真值文本同源，
  且经交叉核对可用：实测在 8 张图上与 yolo 检测框的 IoU 为 0.83~0.92
  （不满 1 是因为 CCPD 的框更紧、检测框略松，属正常偏差）；
* 子集构成：base(正常) 67 + weather 8 + blur 6 + tilt 6 + challenge 6
  + fn 4 + db 2 + rotate 1，覆盖常规与困难场景。

> 注意：文件名里的**号码下标串用的是另一套字符表**（与该数据集的真值来源不同），
> 所以文本真值只认 ``_labels.json``，文件名只取 bbox / 四角点。

## 指标口径

每个引擎在同一批图上报两组互补指标：

**识别**（以 ``_labels.json`` 的文本为准）

1. **图级严格命中**：真值牌 == 某个检出文本（多检出不扣分，漏检/读错即败）。
   这是最严的指标，也是对外可以引用的那一个；
2. **字符级准确率**：把该图"最接近真值的检出"与真值逐位比较，
   ``1 - 编辑距离 / 真值长度``（无检出记 0），再对图求平均 ——
   它回答"差一个字符"到底占多少；
3. **近失**：编辑距离 = 1 的图数（只差一个字符）；
4. **误检条目**：既不是真值、也不是近似的多余检出数量。

**定位**（以文件名 bbox 为真值，IoU ≥ ``--iou`` 记命中）

* 图级命中率、精确率（命中框 / 全部检出框）、平均最佳 IoU。
  定位指标的价值在于把"没找到牌"和"找到了但读错"分开 ——
  这两类失败要用完全不同的办法治。

**耗时**：中位数与 P95（ms/张）。

用法::

    python tools/eval_ccpd.py                          # 默认只评 deep（向后兼容）
    python tools/eval_ccpd.py --engine yolo            # 单引擎
    python tools/eval_ccpd.py --engines cv,deep,yolo   # 多引擎对比
    python tools/eval_ccpd.py --engine fusion --fusion-strategy chain
    # 一次跑完全部对比（推荐，报告里的表就是这条命令产的）
    python tools/eval_ccpd.py --engines cv,deep,yolo,fusion:ensemble,fusion:chain,fusion:auto \
        --json reports/eval_all.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

LABELS = ROOT / "datasets" / "ccpd_labeled" / "_labels.json"
IMG_DIR = ROOT / "datasets" / "ccpd_labeled"
ENGINE_CHOICES = ("cv", "deep", "yolo", "fusion")
FUSION_STRATEGY_CHOICES = ("ensemble", "chain", "auto")


def parse_spec(raw: str) -> tuple[str, str]:
    """把 ``fusion:chain`` 这样的规格拆成 ``(引擎名, 融合策略)``。

    不带冒号的写法（``deep``）策略为空字符串；``fusion`` 不带策略时由调用方
    补默认值，这样 ``--engines cv,deep,fusion:chain`` 与 ``--engine fusion
    --fusion-strategy chain`` 两种写法等价。
    """
    name, _, strat = raw.partition(":")
    return name.strip(), strat.strip()


def _d1(a: str, b: str) -> bool:
    """编辑距离是否 ≤1（仅用于近失诊断，不等价于识别正确）。"""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1 :] == a for i in range(len(b)))


def _edit_distance(a: str, b: str) -> int:
    """标准编辑距离（车牌长度 ≤8，直接 DP 即可）。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """``(x, y, w, h)`` 框的 IoU。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _diff_pairs(truth: str, pred: str) -> list[tuple[str, str]]:
    """用编辑距离回溯，列出 ``真值 → 检出`` 的逐字符差异。

    返回 ``(真值字符, 检出字符)``；插入记 ``("<ins>", 字符)``，删除记
    ``(字符, "<del>")``。用于统计"形近字"到底混淆了哪几对（如 Z↔2）。
    """
    n, m = len(truth), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (truth[i - 1] != pred[j - 1]),
            )

    out: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (truth[i - 1] != pred[j - 1]):
            if truth[i - 1] != pred[j - 1]:
                out.append((truth[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append((truth[i - 1], "<del>"))
            i -= 1
        else:
            out.append(("<ins>", pred[j - 1]))
            j -= 1
    return list(reversed(out))


# 失败主类：互斥，每张未严格命中的图恰好落进一类。
# 键用稳定的 ASCII（表格脚本按它取数），中文标签只用于显示。
FAIL_KEYS = ("no_box", "no_text", "near", "gross")
FAIL_LABELS = {
    "no_box": "无框（定位失败）",
    "no_text": "有框无字（定位到但没读出来）",
    "near": "形近字（差 1 字符）",
    "gross": "严重错读（差 ≥2 字符）",
}


def classify_failures(payload: dict[str, Any], iou_threshold: float) -> dict[str, Any]:
    """把未命中的图归入互斥主类，并统计形近字混淆对与定位失手率。

    四类互斥，按"卡在哪一环"划分 —— 因为不同环节要用完全不同的办法治：

    * **无框**（``no_box``）：一个框都没出 —— 定位/召回问题，换识别模型无用；
    * **有框无字**（``no_text``）：框定位对了却没读出字符 —— 多半是分割/矫正
      或识别器在低质图上放弃，换定位模型无用；
    * **形近字**（``near``）：读出来了但只差一个字符 —— 纯识别问题，靠更强
      识别权重或后处理纠错治；
    * **严重错读**（``gross``）：读出来但差 ≥2 字符 —— 整牌读错或版式误判。

    另外单独统计 **定位失手**（最佳 IoU < 阈值），与上面四类正交：
    "形近字"那类里也可能夹着"框歪了所以读错"。
    """
    items = payload.get("items", [])
    cats: dict[str, list[dict[str, Any]]] = {key: [] for key in FAIL_KEYS}
    pairs: dict[tuple[str, str], int] = {}
    by_subset: dict[str, dict[str, int]] = {}
    loc_miss: list[dict[str, Any]] = []

    for it in items:
        sub = it.get("subset", "?")
        slot = by_subset.setdefault(sub, {"total": 0, "fail": 0})
        slot["total"] += 1
        if it.get("exact"):
            continue
        slot["fail"] += 1
        gt, preds = it.get("gt", ""), list(it.get("pred") or [])
        loc_ok = it.get("best_iou", 0.0) >= iou_threshold
        if not loc_ok:
            loc_miss.append(it)

        if not preds:
            cats["no_text" if loc_ok else "no_box"].append(it)
            continue
        best = min(preds, key=lambda p: _edit_distance(gt, p))
        dist = _edit_distance(gt, best)
        if dist == 1:
            cats["near"].append(it)
            for a, b in _diff_pairs(gt, best):
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
        else:
            cats["gross"].append(it)

    total_fail = sum(len(v) for v in cats.values())
    return {
        "total_fail": total_fail,
        "cats": cats,
        "pairs": sorted(pairs.items(), key=lambda kv: -kv[1]),
        "loc_miss": loc_miss,
        "by_subset": by_subset,
        "iou_threshold": iou_threshold,
    }


def print_taxonomy(tag: str, taxonomy: dict[str, Any]) -> None:
    """打印单个引擎的失败分类明细。"""
    total = taxonomy["total_fail"]
    if not total:
        print(f"  [{tag}] 没有失败样本")
        return
    print(f"\n  [{tag}] 未命中 {total} 张 —— 主类分布（互斥，按卡在哪一环分）")
    for key in FAIL_KEYS:
        rows = taxonomy["cats"][key]
        print(f"    {FAIL_LABELS[key]}：{len(rows)} 张 ({len(rows) / total:.1%})")
        for r in rows:
            pred = "、".join(r.get("pred") or []) or "无"
            print(
                f"      - {r['name'][:26]:26s} [{r.get('subset', '?')}] GT={r['gt']} → 检出={pred}"
            )
    if taxonomy["pairs"]:
        shown = taxonomy["pairs"][:15]
        text = "  ".join(f"{a}→{b} × {c}" for (a, b), c in shown)
        print(f"    形近字混淆对（真值→检出 × 次数）: {text}")
    threshold = taxonomy["iou_threshold"]
    print(f"    其中定位失手（最佳 IoU < {threshold}）: {len(taxonomy['loc_miss'])} 张")
    for r in taxonomy["loc_miss"]:
        print(
            f"      - {r['name'][:26]:26s} [{r.get('subset', '?')}] "
            f"best_iou={r.get('best_iou')} GT={r['gt']}"
        )


def taxonomy_to_json(taxonomy: dict[str, Any]) -> dict[str, Any]:
    """把分类结果压成可序列化的精简形式（只留文件名，不留整条明细）。"""
    return {
        "total_fail": taxonomy["total_fail"],
        "iou_threshold": taxonomy["iou_threshold"],
        "cats": {k: [x["name"] for x in v] for k, v in taxonomy["cats"].items()},
        "pairs": [[a, b, c] for (a, b), c in taxonomy["pairs"]],
        "loc_miss": [x["name"] for x in taxonomy["loc_miss"]],
        "by_subset": taxonomy["by_subset"],
    }


def gt_box_from_name(orig: str) -> tuple[int, int, int, int] | None:
    """从 CCPD 原始文件名取定位真值，返回 ``(x, y, w, h)``。"""
    from ccpd_labels import parse

    ann = parse(orig)
    if ann is None:
        return None
    x1, y1, x2, y2 = ann.bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def load_items(limit: int, subset: str) -> list[tuple[str, dict[str, Any]]]:
    """读取评测清单（过滤子集与数量）。"""
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    items = sorted(labels.items())
    if subset:
        items = [(k, v) for k, v in items if v.get("subset") == subset]
    if limit:
        items = items[:limit]
    return items


def build_engine(
    name: str,
    fusion_strategy: str | None,
    secondary_rec: str = "",
    clear_first: str | None = None,
):
    """按名字造引擎（fusion 可指定策略，``secondary_rec`` 可开第二识别器）。"""
    from lpr.config import EngineConfig
    from lpr.engines import create_engine

    cfg = EngineConfig()
    if secondary_rec:
        cfg.secondary_rec_model = secondary_rec
    if clear_first:
        cfg.fusion_auto_clear_first = clear_first
    if secondary_rec or clear_first:
        cfg.validate()
    kwargs: dict[str, Any] = {}
    if name == "fusion" and fusion_strategy:
        kwargs["strategy"] = fusion_strategy
    engine = create_engine(name, cfg, **kwargs)
    engine.load()
    return engine


def evaluate_engine(
    engine, items: list[tuple[str, dict[str, Any]]], iou_threshold: float
) -> dict[str, Any]:
    """在给定样本上评测一个引擎实例。"""
    from lpr.utils.image import imread

    n = 0
    n_img_hit = 0  # 严格：文本完全命中
    n_near = 0  # 近失（差 1 字符）
    n_extra = 0  # 误检条目
    char_sum = 0.0  # 字符级准确率累计
    loc_hit = 0  # 定位命中（最佳 IoU ≥ 阈值）
    iou_sum = 0.0
    det_total = 0  # 全部检出框数（算精确率用）
    det_tp = 0  # 其中与真值 IoU ≥ 阈值的个数
    times: list[float] = []
    per_subset: dict[str, list[int]] = {}  # 子集 → [图数, 识别命中, 定位命中]
    fails: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for name, meta in items:
        img = imread(str(IMG_DIR / name))
        if img is None:
            fails.append({"name": name, "reason": "图片读取失败"})
            continue
        gt_list = list(meta["plates"])
        gt = gt_list[0] if gt_list else ""
        gt_box = gt_box_from_name(meta.get("orig", ""))

        t0 = time.perf_counter()
        results = engine.predict(img)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

        got = [r.text for r in results if r.text]
        n += 1
        det_total += len(results)

        # ---- 识别
        exact = gt in got
        best_dist = min((_edit_distance(gt, g) for g in got), default=len(gt) or 1)
        char_acc = 1.0 - best_dist / max(1, len(gt)) if got else 0.0
        char_sum += max(0.0, char_acc)
        near = (not exact) and any(_d1(gt, g) for g in got)
        extras = sum(1 for g in got if not any(_d1(x, g) for x in gt_list))
        n_extra += extras
        if exact:
            n_img_hit += 1
        if near:
            n_near += 1

        # ---- 定位
        best_iou = 0.0
        best_box = None
        for r in results:
            if not r.bbox:
                continue
            value = _iou(gt_box, r.bbox) if gt_box else 0.0
            if value >= iou_threshold:
                det_tp += 1
            if value > best_iou:
                best_iou, best_box = value, r.bbox
        iou_sum += best_iou
        hit_loc = best_iou >= iou_threshold
        if hit_loc:
            loc_hit += 1

        sub = meta.get("subset", "?")
        slot = per_subset.setdefault(sub, [0, 0, 0])
        slot[0] += 1
        slot[1] += int(exact)
        slot[2] += int(hit_loc)

        details.append(
            {
                "name": name,
                "subset": sub,
                "gt": gt,
                "pred": got,
                "exact": exact,
                "near": near,
                "extra": extras,
                "char_acc": round(max(0.0, char_acc), 4),
                "best_iou": round(best_iou, 4),
                "pred_box": list(best_box) if best_box else None,
                "gt_box": list(gt_box) if gt_box else None,
                "ms": round(elapsed, 1),
                "fusion": {
                    k: v
                    for k, v in (results[0].meta.get("fusion", {}) if results else {}).items()
                    if k in ("strategy", "decided", "support", "agreement", "dropped_count")
                },
            }
        )

        if not (exact or near):
            fails.append(
                {
                    "name": name,
                    "subset": sub,
                    "gt": gt,
                    "pred": got or "无",
                    "loc": "命中" if hit_loc else "未命中",
                    "iou": round(best_iou, 3),
                }
            )

    if n == 0:
        return {"n": 0, "fails": [{"name": "-", "reason": "没有可评测样本"}]}

    sorted_t = sorted(times)
    hit_rate = n_img_hit / n
    metrics = {
        "n": n,
        "img_hit": n_img_hit,
        "img_hit_rate": hit_rate,
        "char_acc": char_sum / n,
        "near": n_near,
        "extra": n_extra,
        "loc_hit": loc_hit,
        "loc_hit_rate": loc_hit / n,
        "loc_precision": det_tp / det_total if det_total else 0.0,
        "det_total": det_total,
        "det_tp": det_tp,
        "mean_best_iou": iou_sum / n,
        "ms_median": statistics.median(sorted_t),
        "ms_p95": sorted_t[min(len(sorted_t) - 1, int(len(sorted_t) * 0.95))],
        "per_subset": per_subset,
    }
    precision = metrics["loc_precision"]
    metrics["loc_f1"] = (
        2 * precision * hit_rate / (precision + hit_rate) if (precision + hit_rate) else 0.0
    )
    return {"summary": metrics, "items": details, "fails": fails}


def print_report(tag: str, payload: dict[str, Any]) -> None:
    """打印单个引擎的结果。"""
    s = payload.get("summary")
    if not s:
        print(f"\n==== {tag} ====")
        print("  没有可评测样本")
        for f in payload.get("fails", []):
            print(f"  {f}")
        return

    print(f"\n==== {tag}（{s['n']} 张）====")
    print(
        f"  识别: 严格命中 {s['img_hit']}/{s['n']} = {s['img_hit_rate']:.1%}"
        f" | 字符级准确率 {s['char_acc']:.1%}"
        f" | 近失 {s['near']} | 误检条目 {s['extra']}"
    )
    print(
        f"  定位: 图级命中 {s['loc_hit']}/{s['n']} = {s['loc_hit_rate']:.1%}"
        f" | 精确率 {s['loc_precision']:.1%}（命中框 {s['det_tp']}/{s['det_total']}）"
        f" | F1 {s['loc_f1']:.1%} | 平均最佳 IoU {s['mean_best_iou']:.3f}"
    )
    print(f"  耗时: 中位 {s['ms_median']:.0f} ms | P95 {s['ms_p95']:.0f} ms")

    if len(s["per_subset"]) > 1:
        print("  分子集（图数 | 识别命中 | 定位命中）:")
        for sub, (cnt, hit, loc) in sorted(s["per_subset"].items()):
            print(
                f"    {sub:10s} {cnt:3d} | {hit:3d} ({hit / cnt:5.1%}) | {loc:3d} ({loc / cnt:5.1%})"
            )

    if payload["fails"]:
        print(f"  未命中明细（{len(payload['fails'])} 条）:")
        for f in payload["fails"]:
            print(
                f"    {f['name'][:28]:28s} [{f.get('subset', '?')}] "
                f"GT={f['gt']} 检出={f['pred']} 定位={f.get('loc', '?')}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 张（冒烟用）")
    ap.add_argument("--subset", type=str, default="", help="只评某子集，如 base")
    ap.add_argument("--engine", type=str, default="deep", choices=ENGINE_CHOICES, help="单引擎")
    ap.add_argument(
        "--engines",
        type=str,
        default="",
        help=(
            "多引擎对比，逗号分隔（可选 "
            + ", ".join(ENGINE_CHOICES)
            + "）；fusion 可写成 fusion:ensemble / fusion:chain / fusion:auto"
        ),
    )
    ap.add_argument(
        "--fusion-strategy",
        type=str,
        default="",
        choices=["", *FUSION_STRATEGY_CHOICES],
        help="裸 fusion 的默认策略（留空用配置默认 ensemble）",
    )
    ap.add_argument(
        "--fusion-clear-first",
        type=str,
        default="",
        help="fusion auto 策略在清晰大图上的首选成员（对应 fusion_auto_clear_first）",
    )
    ap.add_argument("--iou", type=float, default=0.5, help="定位匹配 IoU 阈值")
    ap.add_argument(
        "--secondary-rec",
        type=str,
        default="",
        help="开启第二识别模型（低置信交叉验证）的 ONNX 路径，用于 A/B 实验",
    )
    ap.add_argument(
        "--taxonomy",
        action="store_true",
        help="额外打印失败分类（漏检 / 形近字 / 严重错读 + 混淆对）",
    )
    ap.add_argument("--json", type=str, default="", help="把原始结果写成 JSON")
    args = ap.parse_args()

    if not LABELS.is_file():
        print(f"缺少真值文件 {LABELS}")
        return 1

    names = [x.strip() for x in args.engines.split(",") if x.strip()] or [args.engine]
    specs = [parse_spec(x) for x in names]
    unknown = [n for n, _ in specs if n not in ENGINE_CHOICES]
    if unknown:
        print(f"未知引擎: {', '.join(unknown)}（可选 {', '.join(ENGINE_CHOICES)}）")
        return 1
    bad_strategy = [s for n, s in specs if s and s not in FUSION_STRATEGY_CHOICES]
    if bad_strategy:
        print(
            f"未知融合策略: {', '.join(bad_strategy)}（可选 {', '.join(FUSION_STRATEGY_CHOICES)}）"
        )
        return 1

    items = load_items(args.limit, args.subset)
    if not items:
        print("没有可评测样本")
        return 1

    print(f"评测集: datasets/ccpd_labeled（{len(items)} 张）| 定位 IoU 阈值 {args.iou}")
    out: dict[str, Any] = {}
    for name, strat in specs:
        if name == "fusion":
            strat = strat or args.fusion_strategy or "ensemble"
        tag = f"fusion({strat})" if name == "fusion" else name
        if args.secondary_rec:
            tag += "+第二识别器"
        if args.fusion_clear_first and name == "fusion":
            tag += f"@{args.fusion_clear_first}"
        engine = build_engine(
            name, strat or None, args.secondary_rec, args.fusion_clear_first or None
        )
        payload = evaluate_engine(engine, items, args.iou)
        payload["engine"] = name
        if name == "fusion":
            payload["fusion_strategy"] = strat
        print_report(tag, payload)
        taxonomy = classify_failures(payload, args.iou)
        payload["taxonomy"] = taxonomy_to_json(taxonomy)
        if args.taxonomy:
            print_taxonomy(tag, taxonomy)
        out[tag] = payload

    if len(out) > 1:
        print("\n==== 横向对比 ====")
        print(
            "| 引擎 | 识别严格命中 | 字符级准确率 | 定位命中 | 定位精确率 | 平均IoU | 中位ms | 误检 |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for tag, payload in out.items():
            s = payload.get("summary")
            if s:
                print(
                    f"| {tag} | {s['img_hit_rate']:.1%} | {s['char_acc']:.1%} | "
                    f"{s['loc_hit_rate']:.1%} | {s['loc_precision']:.1%} | "
                    f"{s['mean_best_iou']:.3f} | {s['ms_median']:.0f} | {s['extra']} |"
                )

    if args.json:
        dest = Path(args.json)
        if not dest.is_absolute():
            dest = ROOT / dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n原始结果已写入 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
