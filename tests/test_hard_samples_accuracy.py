"""对抗性退化样张的准确率门槛（CI gate）。

与 tests/test_real_samples_accuracy.py 的分工：
* 原样张门槛守"干净图上不许回退"（命中零容忍）；
* 本门槛守"困难条件下不许变得更差"——用 tools/make_hard_samples.py
  的确定性退化（模糊/缩放/亮度/噪声/倾斜/压缩）把真值样张实时退化后
  跑完整 deep 引擎，只卡**命中率比率**，不卡单张（困难样本存在合理失败）。

退化在内存中完成，仓库不落二进制；全部操作确定性（固定种子），
跨运行结果一致。

阈值说明：按当前实测水平设定——默认配置（无第二模型）47.8%，
启用第二模型（secondary_rec_model）后更高。CI 跑默认配置，门槛取
0.45 留安全边距；已暴露的问题簇（川AA16U52 U→0、京E51619 丢 E、
红尾字退化翻转、琼→京/豫、double_yellow 退化后 OCR 读错）修复后
应相应上调本阈值。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # tools/ 不在包内

pytest.importorskip("rapidocr_onnxruntime")
pytestmark = pytest.mark.slow

#: 对抗样张命中率门槛（默认配置当前实测 47.8%，问题簇修复后应上调）
_THRESHOLD = 0.45


def _degradation_kinds() -> tuple[str, ...]:
    """从退化模块取全部退化类型键，避免两处硬编码漂移。"""
    from tools.make_hard_samples import DEGRADATIONS

    return tuple(sorted(DEGRADATIONS))


_KINDS = _degradation_kinds()

#: 各退化类型的命中统计：kind → (命中数, 计分数, 未命中明细)。
#: 之所以拆成"按类型分测 + 末尾汇总断言"：单测一次跑完 77 张退化图时，多数
#: 样本走最慢的全图 OCR 兜底（单张 0.3~1s），在较慢的 Python/机器上会超过
#: ``pytest --timeout=300`` 的单测上限而被判失败（实测 3.12 下超时、3.13 下
#: 约 99s 侥幸通过）。拆开后每个测试只跑约 11 张，稳定低于上限；汇总断言仍
#: 保留"总命中率"这一语义，且分类型明细便于定位是哪类退化在回退。
_STATS: dict[str, tuple[int, int, list[str]]] = {}


@pytest.fixture(scope="module")
def engine():
    from lpr.config import EngineConfig
    from lpr.engines.deep_engine import DeepEngine

    eng = DeepEngine(EngineConfig())
    eng.load()
    return eng


def _iter_kind(kind: str):
    """只产出某一退化类型的 (条目, 退化图)；退化计算便宜，重的是 engine.predict。"""
    from tools.make_hard_samples import iter_hard_samples

    for entry, img in iter_hard_samples():
        if entry["path"].endswith(f"__{kind}"):
            yield entry, img


@pytest.mark.parametrize("kind", _KINDS)
def test_hard_samples_by_kind(engine, kind):
    """单一退化类型下的命中情况；此处只统计，阈值统一在汇总测试里卡。"""
    n_required = 0
    n_hits = 0
    misses: list[str] = []

    for entry, img in _iter_kind(kind):
        results = engine.predict(img)
        got = {r.text for r in results if r.text}
        expected = set(entry["plates"])

        if entry.get("known_failure"):
            continue  # 已知信息不足的组合不参与门槛

        n_required += 1
        if expected <= got:
            n_hits += 1
        else:
            misses.append(f"{entry['path']}: 期望 {sorted(expected)}，实得 {sorted(got)}")

    _STATS[kind] = (n_hits, n_required, misses)


def test_hard_samples_gate():
    """汇总各退化类型的命中率并卡门槛（依赖上面的分类型测试先行执行）。"""
    assert set(_STATS) == set(_KINDS), "分类型测试未全部执行，无法汇总命中率"

    n_hits = sum(hits for hits, _, _ in _STATS.values())
    n_required = sum(required for _, required, _ in _STATS.values())
    misses = [m for _, _, ms in _STATS.values() for m in ms]

    rate = n_hits / n_required if n_required else 1.0
    per_kind = "，".join(f"{k} {h}/{r}" for k, (h, r, _) in sorted(_STATS.items()))
    detail = (
        f"\n对抗命中率 {n_hits}/{n_required} = {rate:.1%}（门槛 {_THRESHOLD:.0%}）\n"
        f"分类型：{per_kind}"
    )
    if misses:
        detail += "\n未命中:\n  " + "\n  ".join(misses)

    assert rate >= _THRESHOLD, f"对抗样张命中率低于门槛（困难场景识别回退）：{detail}"
