"""三合一引擎（``fusion``）测试。

分三层：

1. **纯函数**（``blur_score`` / ``select_single_engine`` / ``group_results`` /
   ``vote_group`` / ``fuse_results``）—— 不需要任何真引擎，覆盖融合数学。
2. **引擎编排**（用 ``member_factory`` 注入假成员）—— 覆盖三种策略的路由、
   成员退席、异常隔离、生命周期。
3. **注册与配置** —— 保证 GUI/CLI 能选到它、配置字段真的接得上。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lpr.config import (  # noqa: E402
    ENGINE_CHOICES,
    FUSION_MEMBER_CHOICES,
    FUSION_STRATEGIES,
    Config,
    EngineConfig,
)
from lpr.engines import (  # noqa: E402
    EngineUnavailable,
    FusionEngine,
    available_engines,
    create_engine,
    describe_engines,
)
from lpr.engines.fusion_engine import (  # noqa: E402
    DEFAULT_ENGINE_WEIGHTS,
    STRATEGY_AUTO,
    STRATEGY_CHAIN,
    STRATEGY_ENSEMBLE,
    blur_score,
    fuse_group,
    fuse_results,
    group_results,
    select_single_engine,
    validate_members,
    vote_group,
)
from lpr.engines.registry import _REGISTRY  # noqa: E402
from lpr.types import PlateColor, PlateLayout, PlateResult  # noqa: E402


def res(engine: str, text: str, conf: float = 1.0, bbox=(100, 100, 200, 60)) -> PlateResult:
    """造一条成员引擎的输出。"""
    return PlateResult(
        text=text,
        color=PlateColor.BLUE,
        layout=PlateLayout.SINGLE,
        bbox=tuple(bbox) if bbox else None,
        confidence=conf,
        engine=engine,
    )


def sharp_image(shape=(480, 640)) -> np.ndarray:
    """高频噪声图 —— 拉普拉斯方差大，用来模拟"清晰"。"""
    rng = np.random.default_rng(20261015)
    return rng.integers(0, 256, (*shape, 3), dtype=np.uint8)


def blurry_image(shape=(480, 640)) -> np.ndarray:
    """纯色图 —— 拉普拉斯方差为 0，用来模拟"糊"。"""
    return np.full((*shape, 3), 128, np.uint8)


class FakeMember:
    """假成员引擎：只回放预设结果，并记录自己被调了几次。"""

    def __init__(self, name: str, results=None, *, error: Exception | None = None):
        self.name = name
        self._results = list(results or [])
        self._error = error
        self.calls = 0
        self.closed = False
        self.loaded = False

    def ensure_ready(self) -> None:
        self.loaded = True

    def predict(self, image) -> list[PlateResult]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return list(self._results)

    def close(self) -> None:
        self.closed = True


def build(members: dict[str, FakeMember], **kwargs) -> FusionEngine:
    """用假成员造一个融合引擎（成员键顺序即尝试顺序）。"""
    ordered = list(members)
    engine = FusionEngine(
        members=ordered,
        member_factory=lambda name: members[name],
        **kwargs,
    )
    engine.load()
    return engine


# --------------------------------------------------------------------- 模糊度


class TestBlurScore:
    def test_noise_is_sharper_than_flat_color(self):
        assert blur_score(sharp_image()) > blur_score(blurry_image())

    def test_flat_color_is_near_zero(self):
        assert blur_score(blurry_image()) < 1.0

    def test_empty_image_is_zero_not_exception(self):
        assert blur_score(np.zeros((0, 0, 3), np.uint8)) == 0.0

    def test_big_image_is_downscaled_but_still_ordered(self):
        """大图先缩再算 —— 结论不能因为缩了而反过来。"""
        big_sharp = sharp_image((3000, 4000))
        big_flat = blurry_image((3000, 4000))
        assert blur_score(big_sharp) > blur_score(big_flat)


# --------------------------------------------------------------------- 特征选优


class TestSelectSingleEngine:
    def test_small_image_prefers_detector(self):
        pick, why = select_single_engine(sharp_image((300, 300)), ("yolo", "deep", "cv"))
        assert pick == "yolo"
        assert "检测网络" in why

    def test_blurry_image_prefers_detector(self):
        pick, _ = select_single_engine(blurry_image((1000, 1200)), ("yolo", "deep", "cv"))
        assert pick == "yolo"

    def test_large_sharp_image_prefers_deep(self):
        pick, why = select_single_engine(sharp_image((1000, 1400)), ("yolo", "deep", "cv"))
        assert pick == "deep"
        assert "深度学习" in why

    def test_only_available_members_are_considered(self):
        """``deep`` 缺席时，大图清晰场景要顺延给 yolo 而不是挑一个没在岗的。"""
        pick, _ = select_single_engine(sharp_image((1000, 1400)), ("yolo", "cv"))
        assert pick == "yolo"

    def test_single_member_is_returned_as_is(self):
        pick, _ = select_single_engine(sharp_image((300, 300)), ("cv",))
        assert pick == "cv"

    def test_empty_members_raise(self):
        with pytest.raises(ValueError):
            select_single_engine(sharp_image(), ())


# --------------------------------------------------------------------- 分组


class TestGroupResults:
    def test_same_plate_from_two_engines_lands_in_one_group(self):
        groups = group_results([res("deep", "京A12345"), res("yolo", "京A12345")])
        assert len(groups) == 1

    def test_slightly_offset_boxes_are_one_group(self):
        """框有点位移不影响归组 —— 三个引擎的框本来就不会完全重合。"""
        groups = group_results(
            [
                res("deep", "京A12345", bbox=(100, 100, 200, 60)),
                res("cv", "京A12345", bbox=(112, 96, 196, 64)),
            ]
        )
        assert len(groups) == 1

    def test_two_far_apart_plates_stay_separate(self):
        groups = group_results(
            [
                res("deep", "京A12345", bbox=(10, 10, 200, 60)),
                res("deep", "沪B67890", bbox=(900, 700, 200, 60)),
            ]
        )
        assert len(groups) == 2

    def test_results_without_bbox_group_by_identical_text_only(self):
        groups = group_results(
            [res("deep", "京A12345", bbox=None), res("cv", "京A12345", bbox=None)]
        )
        assert len(groups) == 1

    def test_results_without_bbox_and_different_text_never_merge(self):
        """没有框就没有空间依据，宁可不合并也不要两块车牌揉成一团。"""
        groups = group_results(
            [res("deep", "京A12345", bbox=None), res("cv", "沪B67890", bbox=None)]
        )
        assert len(groups) == 2

    def test_empty_text_rows_still_group_spatially(self):
        """检到了但没读出号码的候选也要跟着同组的框走。"""
        groups = group_results([res("yolo", "", bbox=(100, 100, 200, 60)), res("deep", "京A12345")])
        assert len(groups) == 1
        assert len(groups[0]) == 2


# --------------------------------------------------------------------- 投票


class TestVoteGroup:
    def test_majority_wins(self):
        group = [res("deep", "京A12345"), res("yolo", "京A12345"), res("cv", "京A99999")]
        winner, votes = vote_group(group, DEFAULT_ENGINE_WEIGHTS, ["yolo", "deep", "cv"])
        assert winner == "京A12345"
        assert votes["京A12345"] == pytest.approx(2.0)
        assert votes["京A99999"] == pytest.approx(0.6)

    def test_engine_with_lower_weight_loses_even_with_more_candidates(self):
        """一个引擎对同一块车牌给多个候选时**只算一票** —— 话多不等于对。"""
        group = [
            res("deep", "京A12345"),
            res("cv", "京A11111", 0.9),
            res("cv", "京A22222", 0.8),
            res("cv", "京A33333", 0.7),
        ]
        winner, votes = vote_group(group, DEFAULT_ENGINE_WEIGHTS, ["yolo", "deep", "cv"])
        assert winner == "京A12345"
        assert len(votes) == 2

    def test_higher_confidence_member_wins_when_weights_tie(self):
        group = [res("deep", "京A12345", 0.5), res("yolo", "京B67890", 0.95)]
        winner, _ = vote_group(group, DEFAULT_ENGINE_WEIGHTS, ["yolo", "deep", "cv"])
        assert winner == "京B67890"

    def test_zero_confidence_still_counts(self):
        """有的引擎会给 text 但 confidence=0，不能因此让它白跑一趟。"""
        group = [res("deep", "京A12345", 0.0), res("cv", "京A99999", 0.9)]
        winner, votes = vote_group(group, {"deep": 1.0, "cv": 1.0}, ["deep", "cv"])
        assert winner in votes

    def test_tie_is_broken_by_plate_validity_then_order(self):
        """票数打平时：先看号码合法性，再看成员优先级 —— 结果必须确定。"""
        group = [res("yolo", "京A12345", 0.8), res("deep", "京A1234", 0.8)]
        winner, _ = vote_group(group, {"yolo": 1.0, "deep": 1.0}, ["yolo", "deep", "cv"])
        assert winner == "京A12345"

    def test_no_text_means_no_votes(self):
        winner, votes = vote_group([res("yolo", "")], DEFAULT_ENGINE_WEIGHTS, ["yolo"])
        assert winner == ""
        assert votes == {}


# --------------------------------------------------------------------- 单组合成


class TestFuseGroup:
    def test_meta_carries_evidence(self):
        group = [res("deep", "京A12345"), res("yolo", "京A12345"), res("cv", "京A99999")]
        merged = fuse_group(
            group, DEFAULT_ENGINE_WEIGHTS, ["yolo", "deep", "cv"], strategy="ensemble"
        )
        info = merged.meta["fusion"]
        assert merged.engine == "fusion"
        assert merged.text == "京A12345"
        assert info["agreement"] == pytest.approx(2.0 / 2.6, rel=1e-3)
        assert info["support"] == 2
        assert set(info["contributors"]) == {"deep", "yolo", "cv"}
        assert info["contributors"]["cv"]["text"] == "京A99999"

    def test_confidence_is_the_winner_raw_confidence(self):
        """融合置信度必须还是"某个引擎的置信度" —— 否则下游阈值全要重调。"""
        group = [res("deep", "京A12345", 0.87), res("cv", "京A99999", 0.99)]
        merged = fuse_group(
            group, DEFAULT_ENGINE_WEIGHTS, ["yolo", "deep", "cv"], strategy="ensemble"
        )
        assert merged.text == "京A12345"
        assert merged.confidence == pytest.approx(0.87)

    def test_box_is_taken_from_the_winner(self):
        group = [
            res("deep", "京A12345", 0.9, bbox=(10, 20, 100, 40)),
            res("cv", "京A99999", 0.9, bbox=(900, 900, 100, 40)),
        ]
        merged = fuse_group(
            group, DEFAULT_ENGINE_WEIGHTS, ["yolo", "deep", "cv"], strategy="ensemble"
        )
        assert merged.bbox == (10, 20, 100, 40)

    def test_all_empty_text_keeps_box_and_marks_ocr_failed(self):
        merged = fuse_group(
            [res("yolo", "")], DEFAULT_ENGINE_WEIGHTS, ["yolo"], strategy="ensemble"
        )
        assert merged.text == ""
        assert merged.bbox == (100, 100, 200, 60)
        assert merged.meta["ocr_failed"] is True


# --------------------------------------------------------------------- 准入


class TestAdmission:
    def test_single_cv_result_is_dropped_by_default(self):
        """cv 独有结果默认不留 —— 实测 cv 在真实大图上有大量误检。"""
        dropped: list[dict] = []
        out = fuse_results(
            [res("cv", "川J111WE")],
            DEFAULT_ENGINE_WEIGHTS,
            members=("yolo", "deep", "cv"),
            dropped=dropped,
        )
        assert out == []
        assert dropped[0]["text"] == "川J111WE"

    def test_single_trusted_engine_result_survives(self):
        """权重 1.0 的成员可以独存 —— 否则 yolo 独有的远景小牌会被误杀。"""
        out = fuse_results(
            [res("yolo", "京PAP315")], DEFAULT_ENGINE_WEIGHTS, members=("yolo", "deep", "cv")
        )
        assert [r.text for r in out] == ["京PAP315"]

    def test_cv_result_survives_with_second_engine_backing(self):
        out = fuse_results(
            [res("cv", "京A12345"), res("deep", "京A12345")],
            DEFAULT_ENGINE_WEIGHTS,
            members=("yolo", "deep", "cv"),
        )
        assert [r.text for r in out] == ["京A12345"]

    def test_min_support_one_keeps_everything(self):
        out = fuse_results(
            [res("cv", "川J111WE")],
            DEFAULT_ENGINE_WEIGHTS,
            members=("yolo", "deep", "cv"),
            min_support=1,
        )
        assert [r.text for r in out] == ["川J111WE"]

    def test_single_member_relaxes_min_support(self):
        """只有一个成员在岗时"两票"永远凑不齐 —— 必须自动放宽，否则全滤空。"""
        out = fuse_results(
            [res("cv", "京A12345")], DEFAULT_ENGINE_WEIGHTS, members=("cv",), min_support=2
        )
        assert [r.text for r in out] == ["京A12345"]

    def test_empty_text_groups_are_never_dropped(self):
        out = fuse_results([res("cv", "")], DEFAULT_ENGINE_WEIGHTS, members=("yolo", "deep", "cv"))
        assert len(out) == 1
        assert out[0].text == ""

    def test_dropped_list_is_empty_when_nothing_filtered(self):
        dropped: list[dict] = []
        fuse_results(
            [res("yolo", "京A12345")], DEFAULT_ENGINE_WEIGHTS, members=("yolo",), dropped=dropped
        )
        assert dropped == []


# --------------------------------------------------------------------- 编排：并联


class TestEnsembleStrategy:
    def test_all_members_run_and_results_are_fused(self):
        members = {
            "yolo": FakeMember("yolo", [res("yolo", "京A12345", 0.9)]),
            "deep": FakeMember("deep", [res("deep", "京A12345", 0.9)]),
            "cv": FakeMember("cv", [res("cv", "京A12345", 0.9)]),
        }
        engine = build(members, strategy=STRATEGY_ENSEMBLE)
        out = engine.predict(sharp_image())
        assert all(m.calls == 1 for m in members.values())
        assert [r.text for r in out] == ["京A12345"]
        assert out[0].meta["fusion"]["support"] == 3
        assert out[0].engine == "fusion"

    def test_elapsed_is_recorded_per_member(self):
        members = {"yolo": FakeMember("yolo", [res("yolo", "京A12345")])}
        engine = build(members, strategy=STRATEGY_ENSEMBLE)
        out = engine.predict(sharp_image())
        assert set(out[0].meta["fusion"]["elapsed_ms"]) == {"yolo"}

    def test_one_member_crashing_does_not_take_down_the_rest(self):
        members = {
            "yolo": FakeMember("yolo", error=RuntimeError("模型爆了")),
            "deep": FakeMember("deep", [res("deep", "京A12345")]),
        }
        engine = build(members, strategy=STRATEGY_ENSEMBLE)
        out = engine.predict(sharp_image())
        assert [r.text for r in out] == ["京A12345"]
        assert "yolo" in engine.member_errors
        assert out[0].meta["fusion"]["skipped"]["yolo"].startswith("RuntimeError")

    def test_dropped_candidates_are_reported(self):
        members = {
            "cv": FakeMember("cv", [res("cv", "川J111WE", bbox=(10, 10, 100, 40))]),
            "deep": FakeMember("deep", [res("deep", "京A12345", bbox=(900, 900, 100, 40))]),
        }
        engine = build(members, strategy=STRATEGY_ENSEMBLE)
        out = engine.predict(sharp_image())
        assert [r.text for r in out] == ["京A12345"]
        assert [d["text"] for d in engine.last_dropped] == ["川J111WE"]
        assert out[0].meta["fusion"]["dropped_count"] == 1

    def test_no_results_at_all_returns_empty(self):
        members = {"deep": FakeMember("deep", [])}
        engine = build(members, strategy=STRATEGY_ENSEMBLE)
        assert engine.predict(sharp_image()) == []
        assert engine.last_dropped == []


# --------------------------------------------------------------------- 编排：链式


class TestChainStrategy:
    def test_confident_first_member_stops_the_chain(self):
        members = {
            "yolo": FakeMember("yolo", [res("yolo", "京A12345", 0.95)]),
            "deep": FakeMember("deep", [res("deep", "沪B67890", 0.99)]),
            "cv": FakeMember("cv", [res("cv", "粤C11111", 0.99)]),
        }
        engine = build(members, strategy=STRATEGY_CHAIN)
        out = engine.predict(sharp_image())
        assert [r.text for r in out] == ["京A12345"]
        assert members["yolo"].calls == 1
        assert members["deep"].calls == 0 and members["cv"].calls == 0
        assert out[0].meta["fusion"]["decided"] == "yolo"

    def test_weak_result_falls_through_to_next_member(self):
        members = {
            "yolo": FakeMember("yolo", [res("yolo", "京A12345", 0.4)]),
            "deep": FakeMember("deep", [res("deep", "京A12345", 0.9)]),
            "cv": FakeMember("cv", []),
        }
        engine = build(members, strategy=STRATEGY_CHAIN)
        out = engine.predict(sharp_image())
        assert out[0].confidence == pytest.approx(0.9)
        assert out[0].meta["fusion"]["tried"] == ["yolo", "deep"]
        assert out[0].meta["fusion"]["decided"] == "deep"

    def test_all_weak_returns_the_best_effort(self):
        members = {
            "yolo": FakeMember("yolo", [res("yolo", "京A12345", 0.4)]),
            "deep": FakeMember("deep", []),
            "cv": FakeMember("cv", [res("cv", "京A12345", 0.6)]),
        }
        engine = build(members, strategy=STRATEGY_CHAIN)
        out = engine.predict(sharp_image())
        assert out[0].confidence == pytest.approx(0.6)
        assert out[0].meta["fusion"]["decided"] == "best-effort"

    def test_more_plates_beats_higher_confidence_in_best_effort(self):
        """兜底比较先看"读出几个"，再看置信度 —— 召回优先。"""
        members = {
            "yolo": FakeMember(
                "yolo",
                [
                    res("yolo", "京A12345", 0.5),
                    res("yolo", "沪B67890", 0.5, bbox=(800, 600, 100, 40)),
                ],
            ),
            "deep": FakeMember("deep", [res("deep", "粤C11111", 0.6)]),
            "cv": FakeMember("cv", []),
        }
        engine = build(members, strategy=STRATEGY_CHAIN)
        out = engine.predict(sharp_image())
        assert len(out) == 2, "两条低置信度的结果应该赢过一条高置信度的"
        assert {r.text for r in out} == {"京A12345", "沪B67890"}
        assert out[0].meta["fusion"]["decided"] == "best-effort"

    def test_all_members_empty_returns_empty(self):
        members = {"yolo": FakeMember("yolo", []), "deep": FakeMember("deep", [])}
        engine = build(members, strategy=STRATEGY_CHAIN)
        assert engine.predict(sharp_image()) == []

    def test_member_error_is_skipped_not_fatal(self):
        members = {
            "yolo": FakeMember("yolo", error=RuntimeError("boom")),
            "deep": FakeMember("deep", [res("deep", "京A12345", 0.9)]),
        }
        engine = build(members, strategy=STRATEGY_CHAIN)
        out = engine.predict(sharp_image())
        assert [r.text for r in out] == ["京A12345"]
        assert out[0].meta["fusion"]["tried"] == ["yolo", "deep"]


# --------------------------------------------------------------------- 编排：auto


class TestAutoStrategy:
    def test_small_image_routes_to_detector(self):
        members = {
            "yolo": FakeMember("yolo", [res("yolo", "京A12345")]),
            "deep": FakeMember("deep", [res("deep", "沪B67890")]),
        }
        engine = build(members, strategy=STRATEGY_AUTO)
        out = engine.predict(sharp_image((240, 320)))
        assert out[0].text == "京A12345"
        assert members["deep"].calls == 0
        assert "检测网络" in out[0].meta["fusion"]["reason"]

    def test_large_sharp_image_routes_to_deep(self):
        members = {
            "yolo": FakeMember("yolo", [res("yolo", "京A12345")]),
            "deep": FakeMember("deep", [res("deep", "沪B67890")]),
        }
        engine = build(members, strategy=STRATEGY_AUTO)
        out = engine.predict(sharp_image((1000, 1400)))
        assert out[0].text == "沪B67890"
        assert members["yolo"].calls == 0

    def test_empty_first_member_falls_through(self):
        members = {
            "yolo": FakeMember("yolo", []),
            "deep": FakeMember("deep", [res("deep", "沪B67890")]),
            "cv": FakeMember("cv", [res("cv", "粤C11111")]),
        }
        engine = build(members, strategy=STRATEGY_AUTO)
        out = engine.predict(sharp_image((240, 320)))
        assert out[0].text == "沪B67890"
        assert out[0].meta["fusion"]["tried"] == ["yolo", "deep"]

    def test_no_member_hits_returns_empty(self):
        members = {"yolo": FakeMember("yolo", []), "deep": FakeMember("deep", [])}
        engine = build(members, strategy=STRATEGY_AUTO)
        assert engine.predict(sharp_image((240, 320))) == []


# --------------------------------------------------------------------- 生命周期


class TestLifecycle:
    def test_unavailable_member_retires_without_breaking_others(self):
        def factory(name: str):
            if name == "yolo":
                raise EngineUnavailable("没有权重")
            return FakeMember(name, [res(name, "京A12345")])

        engine = FusionEngine(members=("yolo", "deep"), member_factory=factory)
        engine.load()
        assert "yolo" in engine.member_errors
        assert set(engine._members) == {"deep"}
        assert [r.text for r in engine.predict(sharp_image())] == ["京A12345"]

    def test_all_members_unavailable_raises_with_reasons(self):
        def factory(name: str):
            raise EngineUnavailable(f"{name} 缺依赖")

        engine = FusionEngine(members=("yolo", "deep"), member_factory=factory)
        with pytest.raises(EngineUnavailable) as exc:
            engine.load()
        assert "yolo" in str(exc.value) and "deep" in str(exc.value)

    def test_close_does_not_close_shared_members(self):
        """成员是共享单例 —— 关掉它们会顺手把别的引擎砸了。"""
        member = FakeMember("deep", [res("deep", "京A12345")])
        engine = build({"deep": member})
        engine.close()
        assert member.closed is False
        assert engine._members == {}

    def test_predict_ensures_ready_on_first_use(self):
        member = FakeMember("deep", [res("deep", "京A12345")])
        engine = FusionEngine(members=("deep",), member_factory=lambda name: member)
        engine.predict(sharp_image())  # 不显式 load
        assert member.loaded is True

    def test_path_input_is_accepted(self, tmp_path):
        import cv2

        path = tmp_path / "plate.jpg"
        cv2.imwrite(str(path), sharp_image((240, 320)))
        member = FakeMember("deep", [res("deep", "京A12345")])
        engine = build({"deep": member})
        out = engine.predict(path)
        assert [r.text for r in out] == ["京A12345"]


# --------------------------------------------------------------------- 参数校验


class TestMemberValidation:
    def test_unknown_member_is_rejected(self):
        with pytest.raises(ValueError, match="未知融合成员"):
            validate_members(("deep", "不存在的引擎"))

    def test_self_and_strategy_names_are_rejected(self):
        for bad in ("fusion", "cascade", "auto"):
            with pytest.raises(ValueError, match="不能作为融合成员"):
                validate_members((bad,))

    def test_duplicate_member_is_rejected(self):
        with pytest.raises(ValueError, match="重复"):
            validate_members(("deep", "deep"))

    def test_empty_member_list_is_rejected(self):
        with pytest.raises(ValueError, match="不能为空"):
            validate_members(())

    def test_unknown_strategy_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="未知融合策略"):
            FusionEngine(strategy="乱写的")

    def test_member_choices_constant_matches_registry(self):
        """``config.FUSION_MEMBER_CHOICES`` 是给 config 层用的（不能 import engines），
        必须和真实注册表保持一致 —— 这条断言就是两者的同步钩子。
        """
        concrete = set(available_engines()) - {"fusion"}
        assert set(FUSION_MEMBER_CHOICES) == concrete


# --------------------------------------------------------------------- 注册与配置


class TestRegistrationAndConfig:
    def test_engine_is_registered_and_selectable(self):
        assert "fusion" in available_engines()
        assert "fusion" in ENGINE_CHOICES
        engine = create_engine("fusion", None)
        assert isinstance(engine, FusionEngine)
        assert engine.name == "fusion"

    def test_describe_engines_renders_for_gui(self):
        info = next(e for e in describe_engines() if e["name"] == "fusion")
        assert info["display_name"] == "三合一引擎"
        assert info["description"]
        assert isinstance(info["available"], bool)
        assert info["reason"]

    def test_cli_accepts_fusion(self):
        from lpr.cli import _make_parser

        parser = _make_parser()
        args = parser.parse_args(["predict", "samples", "--engine", "fusion"])
        assert args.engine == "fusion"

    def test_default_config_yields_ensemble(self):
        assert Config().engine.fusion_strategy == "ensemble"

    def test_config_fields_plumb_into_engine(self):
        cfg = EngineConfig(
            fusion_strategy="chain",
            fusion_members=("deep", "cv"),
            fusion_engine_weights={"deep": 1.0, "cv": 0.5},
            fusion_group_iou=0.5,
            fusion_chain_accept=0.8,
            fusion_min_support=1,
            fusion_trusted_weight=0.9,
        )
        engine = create_engine("fusion", cfg)
        assert engine.strategy == "chain"
        assert engine.members == ("deep", "cv")
        assert engine.engine_weights == {"deep": 1.0, "cv": 0.5}
        assert engine.group_iou == pytest.approx(0.5)
        assert engine.chain_accept == pytest.approx(0.8)
        assert engine.min_support == 1
        assert engine.trusted_weight == pytest.approx(0.9)

    def test_explicit_kwargs_override_config(self):
        cfg = EngineConfig(fusion_strategy="ensemble", fusion_members=("deep",))
        engine = create_engine("fusion", cfg, strategy="chain", members=["cv"])
        assert engine.strategy == "chain"
        assert engine.members == ("cv",)

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"fusion_strategy": "bogus"}, "fusion_strategy"),
            ({"fusion_members": ("deep", "bogus")}, "fusion_members"),
            ({"fusion_members": ("fusion",)}, "fusion_members"),
            ({"fusion_members": ()}, "fusion_members"),
            ({"fusion_members": ("deep", "deep")}, "fusion_members"),
            ({"fusion_engine_weights": {"deep": 0.0}}, "fusion_engine_weights"),
            ({"fusion_group_iou": 0.0}, "fusion_group_iou"),
            ({"fusion_chain_accept": 1.5}, "fusion_chain_accept"),
            ({"fusion_min_support": 0}, "fusion_min_support"),
            ({"fusion_trusted_weight": -1.0}, "fusion_trusted_weight"),
        ],
    )
    def test_invalid_config_is_rejected(self, kwargs, field):
        with pytest.raises(ValueError, match=field):
            EngineConfig(**kwargs).validate()

    def test_strategy_choices_constant_is_in_sync(self):
        assert set(FUSION_STRATEGIES) == {STRATEGY_ENSEMBLE, STRATEGY_CHAIN, STRATEGY_AUTO}

    def test_registry_has_no_duplicate_names(self):
        """注册表里每个名字都指向各自的类（防止误覆盖）。"""
        assert _REGISTRY["fusion"] is FusionEngine
