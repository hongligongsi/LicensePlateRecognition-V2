# -*- coding: utf-8 -*-
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


@pytest.fixture(scope="module")
def engine():
    from lpr.config import EngineConfig
    from lpr.engines.deep_engine import DeepEngine

    eng = DeepEngine(EngineConfig())
    eng.load()
    return eng


def test_hard_samples_gate(engine):
    from tools.make_hard_samples import iter_hard_samples

    n_required = 0
    n_hits = 0
    misses: list[str] = []

    for entry, img in iter_hard_samples():
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

    rate = n_hits / n_required if n_required else 1.0
    detail = f"\n对抗命中率 {n_hits}/{n_required} = {rate:.1%}（门槛 {_THRESHOLD:.0%}）"
    if misses:
        detail += "\n未命中:\n  " + "\n  ".join(misses)

    assert rate >= _THRESHOLD, f"对抗样张命中率低于门槛（困难场景识别回退）：{detail}"
