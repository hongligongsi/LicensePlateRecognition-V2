"""真实样张准确率门槛（CI gate）。

守什么：
    tests/test_regression_baseline.py 守的是"行为不变"（cv 引擎的候选几何
    快照），本文件守的是"识别效果不退"——对 tests/baselines/ground_truth.json
    里人工标注的真值样张跑完整 deep 引擎，三条门槛：

    1. **命中门槛**：除 known_failure 外的每张样张，真值车牌必须全部命中
       （多牌样张漏掉任何一块即败）。历史缺陷（refine_by_color 过度裁剪
       导致黄牌漏检、粘连双牌整块丢弃）都属于"整块丢失"，此条直接拦截；
    2. **颜色门槛**：命中车牌的底色必须与标注一致，零容忍——黑底白字
       误判危险品、领馆牌底色判错这类历史缺陷都在此拦截；
    3. **比率门槛**：整体命中率 ≥ ground_truth.json 的 _threshold，
       作为前两条之外的聚合双保险（样本集扩大后更细粒度的容错由它兜底）。

怎么跑：
    依赖 rapidocr（deep 引擎），未安装时整个模块 skip——本地最小环境
    不受影响；CI 的 test job 已安装 deep 依赖，门槛在 CI 真实生效。
    加载 OCR 模型 + 识别 11 张样张约 20 s，按仓库约定标记 slow：
    本地 ``pytest -m "not slow"`` 可跳过，CI 全量执行。

改标注：
    known_failure 命中后应移出 known_failure（测试会打印提示）；
    新增样张时同步补标注；真值本身写错了就直接改真值，不要放宽门槛。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# deep 引擎依赖缺失（最小安装）时整模块跳过；CI 装了 deep，不跳
pytest.importorskip("rapidocr_onnxruntime")

pytestmark = pytest.mark.slow

#: 真值标注文件（与测试同目录的 baselines/ 下）
GT_PATH = ROOT / "tests" / "baselines" / "ground_truth.json"
_GT = json.loads(GT_PATH.read_text(encoding="utf-8"))
_SAMPLES: list[dict] = _GT["samples"]
_THRESHOLD = float(_GT.get("_threshold", 0.9))


@pytest.fixture(scope="module")
def engine():
    """模块级共享的 deep 引擎：OCR 模型只加载一次（约 1 s）。"""
    from lpr.config import EngineConfig
    from lpr.engines.deep_engine import DeepEngine

    eng = DeepEngine(EngineConfig())
    # 不做 try/except skip：依赖已装齐还加载失败属于真缺陷，
    # 必须让 CI 红掉，而不是静默跳过把门槛变成摆设
    eng.load()
    return eng


def test_ground_truth_paths_exist():
    """标注里引用的样张必须真实存在——防止改名/移动后门槛悄悄空转。"""
    missing = [s["path"] for s in _SAMPLES if not (ROOT / s["path"]).is_file()]
    assert not missing, f"标注引用的样张不存在: {missing}"


def test_accuracy_gate(engine):
    """三条门槛：命中 + 底色 + 比率，失败消息里给全部明细便于定位。"""
    from lpr.utils.image import imread

    misses: list[str] = []         # 命中门槛：整块车牌丢失
    color_mismatch: list[str] = []  # 颜色门槛：底色判错
    n_required = 0                 # 分母 = 常规样张数（known_failure 不计）
    n_hits = 0

    for s in _SAMPLES:
        img = imread(str(ROOT / s["path"]))
        assert img is not None, f"{s['path']}: 图片读取失败"

        # 只识别一次，号码集合与按号码索引的结果都从同一批结果构造，
        # 避免 deep 引擎的多次全量推理让 CI 时间翻倍
        results = engine.predict(img)
        by_text = {r.text: r for r in results if r.text}
        got = set(by_text)
        expected = set(s["plates"])

        # known_failure：已知失败的样张不参与任何门槛；
        # 若某天能命中了，提示维护者把它转正（不强行失败打扰 CI）
        if s.get("known_failure"):
            if expected <= got:
                print(f"\n[提示] known_failure 已能命中，建议移出: {s['path']}")
            continue

        n_required += 1
        if expected <= got:
            n_hits += 1
            # 颜色门槛：只对真值车牌校验底色（多余检出不管，去重层会合并）
            for plate in expected:
                want = s["color"]
                have = by_text[plate].color.value
                if have != want:
                    color_mismatch.append(
                        f"{s['path']}: {plate} 底色判成 {have}，标注为 {want}"
                    )
        else:
            misses.append(f"{s['path']}: 期望 {sorted(expected)}，实得 {sorted(got)}")

    rate = n_hits / n_required if n_required else 1.0
    detail = f"\n命中率 {n_hits}/{n_required} = {rate:.1%}（门槛 {_THRESHOLD:.0%}）"
    if misses:
        detail += "\n命中失败:\n  " + "\n  ".join(misses)
    if color_mismatch:
        detail += "\n底色不符:\n  " + "\n  ".join(color_mismatch)

    assert not misses, f"有样张整块丢失（识别效果回退）：{detail}"
    assert not color_mismatch, f"有样张底色判错（颜色判定回退）：{detail}"
    assert rate >= _THRESHOLD, f"命中率低于门槛：{detail}"
