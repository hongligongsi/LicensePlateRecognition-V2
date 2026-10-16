"""把 ``tools/eval_ccpd.py`` 产出的 JSON 汇总成 markdown 表格。

这不是一个新的评测器 —— 评测只在 ``eval_ccpd.py`` 里做一次，本脚本只负责
**把同一份 JSON 换个摆法**，好让 ``docs/ACCURACY.md`` 里的每一张表都能由
命令复现，而不是手抄。

用法::

    python tools/eval_ccpd.py \\
        --engines cv,deep,yolo,fusion:ensemble,fusion:chain,fusion:auto \\
        --taxonomy --json reports/eval_all.json
    python tools/eval_report.py reports/eval_all.json

输出四张表 + 两份清单：

1. **总览**：每个配置的识别/定位指标与耗时；
2. **分子集**：按 base / weather / blur / tilt / challenge / fn / db / rotate 拆；
3. **失败分类**：无框 / 有框无字 / 形近字 / 严重错读（互斥），附形近字混淆对；
4. **互补性**：两两之间"只有 A 对"的图数，以及"至少一个对"的 oracle 上界；
5. **硬样本**：所有配置都读错的图。

分类口径与中文标签由 :mod:`eval_ccpd` 单点定义，本脚本只负责排版，
避免两处各写一份中文键而悄悄漂移。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_ccpd import FAIL_KEYS, FAIL_LABELS

SUBSET_ORDER = ["base", "weather", "blur", "tilt", "challenge", "fn", "db", "rotate"]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_engines(out: dict[str, Any]) -> list[str]:
    """只取非 fusion 的单引擎配置。"""
    return [tag for tag in out if not tag.startswith("fusion")]


def table_overview(out: dict[str, Any]) -> str:
    """总览表：识别 + 定位 + 耗时。"""
    lines = [
        "| 配置 | 严格命中 | 字符级准确率 | 近失 | 误检条目 | 定位命中 | 定位精确率 | 定位F1 | 平均IoU | 中位ms | P95ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, payload in out.items():
        s = payload.get("summary")
        if not s:
            lines.append(f"| {tag} | — | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {tag} | {s['img_hit']}/{s['n']} = {s['img_hit_rate']:.1%} "
            f"| {s['char_acc']:.1%} | {s['near']} | {s['extra']} "
            f"| {s['loc_hit_rate']:.1%} | {s['loc_precision']:.1%} "
            f"| {s['loc_f1']:.1%} | {s['mean_best_iou']:.3f} "
            f"| {s['ms_median']:.0f} | {s['ms_p95']:.0f} |"
        )
    return "\n".join(lines)


def table_subsets(out: dict[str, Any]) -> str:
    """分子集表：每个子集一行，每个配置两列（识别命中 / 定位命中）。"""
    tags = list(out)
    subsets = [
        sub
        for sub in SUBSET_ORDER
        if any(sub in (p.get("summary") or {}).get("per_subset", {}) for p in out.values())
    ]

    lines = ["| 子集 | 图数 | " + " | ".join(f"{t} 识别/定位" for t in tags) + " |"]
    lines.append("|---|---|" + "---|" * len(tags))
    for sub in subsets:
        count = 0
        cells = []
        for tag in tags:
            slot = (out[tag].get("summary") or {}).get("per_subset", {}).get(sub)
            if slot:
                count = slot[0]
                cells.append(f"{slot[1]}/{slot[0]} · {slot[2]}/{slot[0]}")
            else:
                cells.append("—")
        lines.append(f"| {sub} | {count} | " + " | ".join(cells) + " |")

    total = (out[tags[0]].get("summary") or {}).get("n", 0) if tags else 0
    cells = []
    for tag in tags:
        s = out[tag].get("summary") or {}
        cells.append(
            f"{s.get('img_hit', 0)}/{s.get('n', 0)} · {s.get('loc_hit', 0)}/{s.get('n', 0)}"
        )
    lines.append(f"| **合计** | {total} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_taxonomy(out: dict[str, Any]) -> str:
    """失败分类表：互斥主类 + 混淆对。

    列数 = 配置 + 未命中 + ``len(FAIL_KEYS)`` + 定位失手 + 混淆对。表头、分隔行与
    数据行三处必须同宽，否则 markdown 表会静默错位 ——
    ``tests/test_eval_ccpd.py`` 用断言钉住了这一点。
    """
    lines = [
        "| 配置 | 未命中 | "
        + " | ".join(FAIL_LABELS[k] for k in FAIL_KEYS)
        + " | 其中定位失手 | 形近字混淆对 |",
        "|---|---|" + "---|" * (len(FAIL_KEYS) + 2),
    ]
    for tag, payload in out.items():
        tx = payload.get("taxonomy")
        if not tx:
            lines.append(f"| {tag} | — |" + " — |" * (len(FAIL_KEYS) + 2))
            continue
        counts = [str(len(tx["cats"].get(k, []))) for k in FAIL_KEYS]
        pairs = "、".join(f"{a}→{b}×{c}" for a, b, c in tx.get("pairs", [])[:8]) or "—"
        lines.append(
            f"| {tag} | {tx['total_fail']} | "
            + " | ".join(counts)
            + f" | {len(tx.get('loc_miss', []))} | {pairs} |"
        )
    return "\n".join(lines)


def _right_sets(out: dict[str, Any]) -> dict[str, set[str]]:
    return {
        tag: {it["name"] for it in p.get("items", []) if it.get("exact")} for tag, p in out.items()
    }


def table_complementarity(out: dict[str, Any]) -> str:
    """互补性：基座引擎两两之间"只有 A 对"的图数，以及 oracle 上界。"""
    sets = _right_sets(out)
    bases = _base_engines(out)
    lines = ["| A \\ B | " + " | ".join(bases) + " |", "|---|" + "---|" * len(bases)]
    for a in bases:
        cells = []
        for b in bases:
            if a == b:
                cells.append(f"{len(sets[a])}（自身）")
            else:
                cells.append(str(len(sets[a] - sets[b])))
        lines.append(f"| **{a}** 独占 | " + " | ".join(cells) + " |")

    all_names = {it["name"] for p in out.values() for it in p.get("items", [])}
    lines.append("")
    if bases:
        union = set().union(*(sets[b] for b in bases))
        every = set.intersection(*(sets[b] for b in bases)) if bases else set()
        lines.append(
            f"- 基座引擎（{'/'.join(bases)}）各自命中："
            + "，".join(f"{b} {len(sets[b])}" for b in bases)
        )
        lines.append(f"- **至少一个命中**（完美路由的上界）：{len(union)}/{len(all_names)}")
        lines.append(f"- **三个都命中**：{len(every)}/{len(all_names)}")
        lines.append(f"- **三个都错**：{len(all_names) - len(union)} 张")
    return "\n".join(lines)


def list_hard(out: dict[str, Any], tags: list[str] | None = None) -> str:
    """所有被选中配置都读错的图。"""
    tags = tags or list(out)
    sets = _right_sets(out)
    all_names = {it["name"] for p in out.values() for it in p.get("items", [])}
    hard = sorted(name for name in all_names if not any(name in sets[t] for t in tags))
    meta: dict[str, dict[str, Any]] = {}
    for p in out.values():
        for it in p.get("items", []):
            meta.setdefault(it["name"], it)
    lines = [f"被 {', '.join(tags)} 全部读错：{len(hard)} 张", ""]
    lines.append("| 图 | 子集 | 真值 |")
    lines.append("|---|---|---|")
    for name in hard:
        it = meta.get(name, {})
        lines.append(f"| {name} | {it.get('subset', '?')} | {it.get('gt', '?')} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=str, help="eval_ccpd.py --json 产出的文件")
    ap.add_argument(
        "--only", type=str, default="", help="只输出指定小节：overview,subsets,taxonomy,comp,hard"
    )
    args = ap.parse_args()

    path = Path(args.json)
    if not path.is_file():
        print(f"找不到 {path}")
        return 1
    out = _load(path)
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}

    def want(key: str) -> bool:
        return not wanted or key in wanted

    if want("overview"):
        print("## 总览\n")
        print(table_overview(out))
        print()
    if want("subsets"):
        print("\n## 分子集\n")
        print(table_subsets(out))
        print()
    if want("taxonomy"):
        print("\n## 失败分类\n")
        print(table_taxonomy(out))
        print()
    if want("comp"):
        print("\n## 互补性\n")
        print(table_complementarity(out))
        print()
    if want("hard"):
        print("\n## 硬样本\n")
        print(list_hard(out))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
