"""deep 引擎投票与融合机制的准确性回归测试。

背景（真实故障，samples/1.jpg）：
    远距离黄牌 ``京E51619``，细笔画的 E 在多个预处理变体中被多数读丢
    （``京51619``），只有少数读对。旧版两个缺陷叠加导致输出 ``京S1619``：

    1. **早停太松**——第 1 轮就锁定，读对的变体根本没跑；
    2. **纠错火上浇油**——``correct('京51619')`` 把数字 5 按"字母被误读"
       修成 S，制造出合法短牌 ``京S1619``，真数字 5 反而被毁；
    3. **整串投票无力回天**——同一错读重复多次每轮单独计票，
       必然碾压仅出现一次的正确读数。

    修复 = 早停加严（跑满广度块 + 领手须为 7/8 位合法牌）+ ROVER 风格逐列
    融合仲裁。本文件用假识别器回放该故障，保证不再复发，并钉住
    :data:`lpr.engines.deep_engine._RECOGNIZE_PLAN`（默认 = ``LOCATE_PLAN``）
    的轮数与早停语义（2026-10-16 换计划时正是这几条断言拦住了"把广度块砍短"
    的改动）。广度块长度不再手写常量，而是从轮次表推导的
    :attr:`RecognizePlan.breadth`，改计划时不会与阈值脱节。

不依赖 OCR 模型，任何机器都能跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lpr.config import EngineConfig  # noqa: E402
from lpr.engines.deep_engine import (  # noqa: E402
    _RECOGNIZE_PLAN,
    DETECT_PLAN,
    DeepEngine,
    _fuse_reads,
)
from lpr.types import PlateColor  # noqa: E402

#: 本文件回放的全部用例都走**默认计划**（没有显式传 ``plan`` 的
#: ``_recognize_candidates`` 调用），即 CV 定位器那条路用的 ``LOCATE_PLAN``。
#: yolo 那条路（``DETECT_PLAN``）的形状约束由 ``test_recognize_plan.py`` 覆盖。
PLAN = _RECOGNIZE_PLAN

# ══════════════════════════════════════════════════ 测试设施


def _make_engine(script: list[tuple[str, float]]) -> tuple[DeepEngine, list[int]]:
    """构造打了桩的 DeepEngine：每次纯识别调用按脚本顺序回放读数。

    返回 ``(engine, 调用次数记录)``，供断言实际跑了几轮。
    """
    eng = DeepEngine(EngineConfig())
    calls: list[int] = []

    def fake_recognize(_img: np.ndarray):
        calls.append(1)
        idx = len(calls) - 1
        if idx >= len(script):
            return []
        text, score = script[idx]
        return [(text, score, [score] * len(text))]

    eng._recognize_only_detailed = fake_recognize  # type: ignore[method-assign]
    return eng, calls


#: 一块足以通过 measure_layout("single") 判定的假车牌 ROI
_FAKE_ROI = np.full((40, 160, 3), 60, dtype=np.uint8)
_FAKE_ROI[10:30, 10:150] = 220


def _recognize_one(eng: DeepEngine):
    results = eng._recognize_candidates([(_FAKE_ROI, PlateColor.BLUE, (0, 0, 160, 40))])
    return results


# ══════════════════════════════════════════════════ _fuse_reads 纯函数


class TestFuseReads:
    def test_recovers_dropped_letter(self):
        """故障回放：5 次丢 E 的读数 + 1 次正确读数 → 融合补回 E。"""
        reads = [
            ("京51619", 0.815),
            ("京51619", 0.810),
            ("京E51619", 0.721),
            ("京51619", 0.814),
            ("京51619", 0.813),
        ]
        assert _fuse_reads(reads) == "京E51619"

    def test_single_read_is_identity(self):
        assert _fuse_reads([("京A12345", 0.9)]) == "京A12345"

    def test_consistent_reads_unchanged(self):
        reads = [("京A12345", 0.9), ("京A12345", 0.88), ("京A12345", 0.91)]
        assert _fuse_reads(reads) == "京A12345"

    def test_weak_insertion_is_rejected(self):
        """插入权重不足参考串分数一半时视为孤证噪声，不补。"""
        reads = [("京51619", 0.9), ("京E51619", 0.3)]
        assert _fuse_reads(reads) == "京51619"

    def test_empty_reads(self):
        assert _fuse_reads([]) is None

    def test_majority_substitution_wins(self):
        """两读数都把第 3 位读成 X（合计票重超过参考串），融合跟随多数。"""
        reads = [("京A1X345", 0.60), ("京A12345", 0.90), ("京A1X345", 0.60)]
        # 参考串 = 京A12345(0.90)；X 的票重 1.2 > 0.9 → 替换
        assert _fuse_reads(reads) == "京A1X345"

    def test_prefix_noise_does_not_break_fusion(self):
        """带前缀噪声的读数不应污染融合结果（插入列会被提取闸门滤掉）。"""
        reads = [("京A12345", 0.9), ("停京A12345", 0.5)]
        fused = _fuse_reads(reads)
        assert fused is not None
        assert "京A12345" in fused


# ══════════════════════════════════════════════════ 投票行为


class TestVotingBehavior:
    def test_real_failure_replays_correctly(self):
        """完整故障回放：6 变体里 5 次丢 E，最终必须输出 京E51619。"""
        script = [
            ("京51619", 0.815),  # (3.0, 0.0) 丢 E
            ("京51619", 0.810),  # (3.0, 0.8) 丢 E
            ("京E51619", 0.721),  # (2.0, 0.0) 读对
            ("京51619", 0.672),  # (2.0, 0.8) 丢 E
            ("京51619", 0.814),  # (4.0, 0.0) 丢 E
            ("京51619", 0.813),  # (4.0, 0.8) 丢 E
        ]
        eng, _calls = _make_engine(script)
        results = _recognize_one(eng)
        assert results, "应产出识别结果"
        assert results[0].text == "京E51619"

    def test_early_stop_waits_for_breadth_block(self):
        """早停必须跑满**广度块**：广度块是"提出不同假设"的最小集合。

        改成"第 1 轮就停"实测会让 yolo 的 CCPD 严格命中从 89 掉回 86 ——
        能推翻错读的 clahe/eq/gch 全排在第 3 轮之后，早停等于把它们全部跳过。
        广度块长度取 :attr:`RecognizePlan.breadth`（同尺度连续轮数），
        不再是手写常量，改计划表时阈值自动跟随。
        """
        script = [("京A12345", 0.95)] * len(PLAN)
        eng, calls = _make_engine(script)
        results = _recognize_one(eng)
        assert results[0].text == "京A12345"
        assert len(calls) == PLAN.breadth, f"应在广度块末尾早停，实际跑了 {len(calls)} 轮"
        assert len(PLAN) > PLAN.breadth, "广度块必须短于整个计划，否则早停没有意义"

    def test_short_fake_plate_never_early_stops(self):
        """6 位"短牌"读数（丢字后被纠错"修"出来的假合法串）必须跑满计划。"""
        script = [("京51619", 0.9)] * len(PLAN)  # 全部丢 E → 全部被纠成 京S1619
        eng, calls = _make_engine(script)
        _recognize_one(eng)
        assert len(calls) == len(PLAN), "短牌领手不允许早停"

    def test_weak_leader_keeps_collecting_into_depth_block(self):
        """领手逐字符置信度低（<0.7）时不允许停，会继续跑深度块；转强后停。"""
        script = (
            [("京A12345", 0.60)] * PLAN.breadth  # 广度块内一直弱
            + [("京A12345", 0.95)]  # 深度块第 1 轮转强
            + [("京A12345", 0.95)] * 3
        )
        eng, calls = _make_engine(script)
        results = _recognize_one(eng)
        assert results[0].text == "京A12345"
        assert len(calls) == PLAN.breadth + 1, f"弱位应越过广度块，实际跑了 {len(calls)} 轮"

    def test_weak_positions_capped_at_max_weak(self):
        """弱位续跑有上限（``plan.max_weak``），不会无限跑下去。"""
        script = [("京A12345", 0.60)] * len(PLAN)
        eng, calls = _make_engine(script)
        results = _recognize_one(eng)
        assert results[0].text == "京A12345"
        assert len(calls) == PLAN.max_weak

    def test_detect_plan_runs_its_full_depth_block_when_weak(self):
        """``DETECT_PLAN`` 的 ``max_weak`` = 整个计划长度：弱位值得把深度块跑完。

        这是两套计划的**证据观差异**所在 —— ``LOCATE_PLAN`` 只换尺度，
        弱位下跑满 6 轮是纯浪费（``max_weak=4``）；``DETECT_PLAN`` 的深度块
        是"换尺度重扫"，弱位读数在另一个尺度上还有救回来的可能。
        """
        assert DETECT_PLAN.max_weak == len(DETECT_PLAN)
        assert _RECOGNIZE_PLAN.max_weak < len(_RECOGNIZE_PLAN)
        script = [("京A12345", 0.60)] * len(DETECT_PLAN)
        eng, calls = _make_engine(script)
        eng._recognize_candidates([(_FAKE_ROI, PlateColor.BLUE, (0, 0, 160, 40))], plan=DETECT_PLAN)
        assert len(calls) == len(DETECT_PLAN)

    def test_fusion_vote_is_recorded_in_variants(self):
        """融合产出的候选应进入投票表（variants 里可见），便于审计。"""
        script = [
            ("京51619", 0.815),
            ("京51619", 0.810),
            ("京E51619", 0.721),
            ("京51619", 0.672),
            ("京51619", 0.814),
            ("京51619", 0.813),
        ]
        eng, _calls = _make_engine(script)
        results = _recognize_one(eng)
        variants = dict(results[0].meta["variants"])
        assert "京E51619" in variants, "融合候选应出现在投票表里"
