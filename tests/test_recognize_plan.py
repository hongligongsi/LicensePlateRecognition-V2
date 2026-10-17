"""识别计划的形状约束（不依赖 OCR 模型）。

2026-10-16 把"尺度 × 锐化"两条轴换成显式计划表（``RecognizePlan``）时，
真正决定准确率的不再是某一行代码，而是**这张表的形状**：

* 广度块必须在**同一尺度**下换算子 —— 它是"提出不同假设"的最小集合；
  对 :data:`DETECT_PLAN` 还要求那几种算子**性质确实不同**（``raw`` +
  ``sharpen`` + ``clahe`` + ``eq`` + ``gch``），砍短它实测会让 yolo 的
  CCPD 严格命中从 89 掉回 86；
* 深度块必须比广度块长，否则早停没有意义；
* 算子里不许有"读起来有用、计划却没用它"的死代码（`close` 就是这样被删掉的）。

v2.8.0 起有了**两套**计划（按 ROI 来源分流），所以形状约束对两套**都**要成立，
并且多一条新的：早停门槛 ``min_stop`` 不得小于广度块长度 —— 否则广度块里
那几种"能推翻错读"的算子会被早停跳过。

这些约束很脆（改一个数字就破坏），所以单独钉一份测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lpr.engines.deep_engine import (  # noqa: E402
    _PREP_OPS,
    _RECOGNIZE_PLAN,
    DETECT_PLAN,
    LOCATE_PLAN,
    RecognizePlan,
    _apply_prep,
)

#: 全部在役计划。``_RECOGNIZE_PLAN`` 与 ``LOCATE_PLAN`` 是同一个对象，
#: 但**分开列出**是有意的：万一有人把默认值改成别的计划，这两条会分叉，
#: 测试要能看见。
PLANS = (LOCATE_PLAN, DETECT_PLAN, _RECOGNIZE_PLAN)
PLAN_IDS = ("locate", "detect", "default")


def _sample_roi() -> np.ndarray:
    """一块有明暗结构的假 ROI（纯色图测不出直方图均衡的效果）。"""
    roi = np.full((40, 160, 3), 60, dtype=np.uint8)
    roi[10:30, 10:150] = 220
    roi[18:24, 40:120] = 30
    return roi


class TestPlanShape:
    @pytest.mark.parametrize("plan", PLANS, ids=PLAN_IDS)
    def test_plan_non_empty(self, plan: RecognizePlan):
        assert plan.rounds

    def test_default_plan_is_the_locate_plan(self):
        """不传计划时必须拿到"deep 引擎的老行为"。

        直接 ``DeepEngine()`` 的调用方（以及 ``deep`` / ``cascade`` 这两条
        走 CV 定位器的路径）靠这一点守住回归基线：默认值一旦被换成
        :data:`DETECT_PLAN`，``deep`` 的 CCPD 命中会掉 1 张、耗时 +50%。
        """
        assert _RECOGNIZE_PLAN is LOCATE_PLAN

    def test_plan_names_are_distinct(self):
        """``plan.name`` 会写进日志用于事后归因，重名就分不清这次跑的哪套。

        ``_RECOGNIZE_PLAN`` 是 ``LOCATE_PLAN`` 的**别名**（同一对象），不算重名，
        所以先按对象身份去重。
        """
        unique = list({id(plan): plan for plan in PLANS}.values())
        names = [plan.name for plan in unique]
        assert len(set(names)) == len(names), f"计划名重复: {names}"
        assert len(unique) == 2, "只应有 locate / detect 两套计划"

    @pytest.mark.parametrize("plan", PLANS, ids=PLAN_IDS)
    def test_every_op_exists(self, plan: RecognizePlan):
        """计划里的算子名必须在算子表里 —— 拼错一个字母就是整轮静默消失。"""
        unknown = {op for _s, op in plan.rounds if op not in _PREP_OPS}
        assert not unknown, f"{plan.name} 计划里出现未知算子: {sorted(unknown)}"

    def test_no_dead_ops(self):
        """算子表里不许有计划没用到的算子（`close` 曾是这样一条死代码）。

        这里要按**两套计划合并**来数：某个算子只被 ``DETECT_PLAN`` 用
        （如 ``clahe`` / ``eq`` / ``gch``）仍然不算死代码。
        """
        listed = set(_PREP_OPS)
        used = {op for plan in PLANS for _s, op in plan.rounds}
        assert listed == used, (
            f"算子表与计划不一致: 表多 {sorted(listed - used)}、计划多 {sorted(used - listed)}"
        )

    @pytest.mark.parametrize("plan", PLANS, ids=PLAN_IDS)
    def test_scales_positive(self, plan: RecognizePlan):
        assert all(scale > 0 for scale, _op in plan.rounds)

    @pytest.mark.parametrize("plan", PLANS, ids=PLAN_IDS)
    def test_breadth_block_is_same_scale(self, plan: RecognizePlan):
        """广度块 = 开头连续同一尺度的那几轮，之后必须换尺度。"""
        scales = [scale for scale, _op in plan.rounds]
        breadth = scales[: plan.breadth]
        assert len(set(breadth)) == 1, f"{plan.name} 的广度块必须是同一尺度: {breadth}"
        assert plan.breadth < len(scales), "广度块之后必须还有轮次（否则深度块为空）"
        assert scales[plan.breadth] != breadth[0], "广度块之后必须换尺度，否则深度块没有意义"

    @pytest.mark.parametrize("plan", PLANS, ids=PLAN_IDS)
    def test_breadth_block_has_no_duplicate_ops(self, plan: RecognizePlan):
        """广度块内不许有重复算子 —— 同一算子再来一遍等于白跑一轮。"""
        ops = [op for _s, op in plan.rounds[: plan.breadth]]
        assert len(set(ops)) == len(ops), f"{plan.name} 广度块内重复算子等于白跑: {ops}"

    def test_detect_plan_breadth_block_is_op_diverse(self):
        """**只有** :data:`DETECT_PLAN` 声称"换假设"，因此它必须真的换。

        ``LOCATE_PLAN`` 的广度块是 ``raw`` / ``sharpen`` 两轮（旧计划遗留的
        "跑两种成像拿一致性证据"），刻意**不**在这里做多样性要求 —— 它赢在
        "换尺度"，而 DETECT_PLAN 赢在"同尺度换性质"。

        这条是**准确率的守门人**：``clahe`` / ``eq`` / ``gch`` 是唯一能推翻
        tilt 子集上"一致地把 D 读成 0"的算子，把广度块收缩成
        ``raw`` / ``sharpen`` 会让 yolo 的 CCPD 严格命中从 89 掉回 86。
        """
        ops = [op for _s, op in DETECT_PLAN.rounds[: DETECT_PLAN.breadth]]
        assert len(set(ops)) >= 3, f"DETECT_PLAN 广度块至少要有 3 种不同算子: {ops}"
        assert {"clahe", "eq", "gch"} <= set(ops), (
            f"能推翻错读的三种增强算子必须在广度块里（它们排在第 3 轮之后，"
            f"广度块收缩就会被早停跳过）: {ops}"
        )

    @pytest.mark.parametrize("plan", PLANS, ids=PLAN_IDS)
    def test_depth_block_is_longer_than_breadth(self, plan: RecognizePlan):
        """早停只在广度块末尾允许 —— 深度块必须更长，否则早停无事可做。"""
        assert len(plan.rounds) > plan.breadth

    @pytest.mark.parametrize("plan", PLANS, ids=PLAN_IDS)
    def test_stop_thresholds_track_the_plan(self, plan: RecognizePlan):
        """三条阈值不变式，改计划时不能忘掉任何一条。"""
        # 1. 早停前必须跑满整个广度块 —— 少跑一轮就有假设没机会出场
        assert plan.min_stop >= plan.breadth, (
            f"{plan.name}: min_stop={plan.min_stop} < 广度块 {plan.breadth}，"
            "早停会把广度块里的算子跳过"
        )
        # 2. 弱位续跑的上限不能低于普通早停门槛（否则弱位反而停得更早）
        assert plan.max_weak >= plan.min_stop
        # 3. 上限不得超过计划长度（否则闸门永远不会触发，成了死条件）
        assert plan.max_weak <= len(plan.rounds)

    @pytest.mark.parametrize("plan", PLANS, ids=PLAN_IDS)
    def test_applies_to_is_documented(self, plan: RecognizePlan):
        """``applies_to`` 是给"这次跑的是哪套计划"做归因的，不能空着。"""
        assert plan.applies_to.strip()

    def test_breadth_property_counts_leading_run(self):
        """``breadth`` 是"开头连续同尺度"的轮数 —— 用合成表把语义钉死。"""
        plan = RecognizePlan(
            name="synthetic",
            rounds=((1.0, "raw"), (1.0, "sharpen"), (2.0, "raw"), (2.0, "sharpen")),
            min_stop=2,
            max_weak=4,
            applies_to="测试用",
        )
        assert plan.breadth == 2
        assert len(plan) == 4

    def test_plan_is_immutable(self):
        """计划是 ``NamedTuple``：拿到它的人不能就地改轮次表。"""
        with pytest.raises(AttributeError):
            LOCATE_PLAN.rounds = ()  # type: ignore[misc]


class TestPrepOps:
    @pytest.mark.parametrize("name", sorted(_PREP_OPS))
    def test_op_returns_bgr_same_size(self, name: str):
        roi = _sample_roi()
        out = _PREP_OPS[name](roi)
        assert out.ndim == 3, f"{name} 应返回三通道（识别后端统一吃 BGR）"
        assert out.shape == roi.shape, f"{name} 不应改变尺寸"
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("name", sorted(_PREP_OPS))
    def test_op_survives_gray_input(self, name: str):
        """灰度 ROI 不能把算子搞崩。

        ``raw`` / ``sharpen`` 会原样透传 2 维图（它们本来就不碰通道），
        其余算子先转灰度再回到三通道 —— 因此这里只要求"不抛异常且不丢尺寸"，
        不强行统一通道数（真实管线给的一律是 BGR）。
        """
        gray = np.full((40, 160), 128, dtype=np.uint8)
        out = _PREP_OPS[name](gray)
        assert out.shape[:2] == gray.shape
        assert out.dtype == np.uint8

    def test_raw_is_identity(self):
        roi = _sample_roi()
        assert _PREP_OPS["raw"](roi) is roi

    def test_apply_prep_upscales(self):
        roi = _sample_roi()
        out = _apply_prep(roi, 3.0, "raw")
        assert out.shape[0] > roi.shape[0] and out.shape[1] > roi.shape[1]

    def test_apply_prep_rejects_unknown_op(self):
        """计划表写错字属编码错误，必须炸出来而不是静默降级。"""
        with pytest.raises(KeyError):
            _apply_prep(_sample_roi(), 3.0, "clah")
