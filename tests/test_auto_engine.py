"""``auto`` 引擎策略：默认按**实测强度**挑最强可用引擎（v2.8.1）。

背景：``engine.default`` 此前写死 ``cascade``（= deep）。而 100 张 CCPD 实测
``yolo`` 89%、``deep`` 79%（``docs/ACCURACY.md`` §2.1）—— 装了 yolo 权重却
不去改配置的用户，白丢 10 个百分点。现在默认值是策略名 ``auto``：

    BEST_ENGINE_ORDER = ("yolo", "deep", "cv")

没有 yolo 权重的机器上 ``yolo.is_available()`` 为 False，``auto`` 安静地落到
``deep``，与旧默认（``cascade``）**行为完全一致** —— 这就是这个改动没有
回归风险的原因：它只在"用户本来就有更强引擎"时才改变结果。

这套测试锁住四件事：

1. 常量与默认值（默认是策略名，不是某个具体引擎）；
2. 解析与退化（yolo → deep → cv → 报错），以及**显式点名不被替换**；
3. ``cascade`` 的语义**没有被顺手改掉**（它永远是 deep 的别名，不会变成 yolo）；
4. 界面下拉框能选中 ``auto`` 且不改写配置 —— v2.7.4 那个坑的翻版防线：
   配置里的值一旦不在下拉框里，就会被静默降级成第一项。
"""

from __future__ import annotations

import logging

import pytest

from lpr.api import Recognizer
from lpr.config import (
    BEST_ENGINE_ORDER,
    CASCADE_RESOLVE_ORDER,
    ENGINE_CHOICES,
    Config,
    EngineConfig,
)
from lpr.engines import ALIAS_ENGINES, EngineUnavailable, describe_engine_choices


def _fake_availability(monkeypatch, table: dict[str, bool]) -> None:
    """把「引擎可用性」换成给的表；表里没出现的引擎 = 未注册（等同不存在）。"""
    fake = {
        name: {
            "name": name,
            "available": ok,
            "reason": "" if ok else "测试里故意置为不可用",
        }
        for name, ok in table.items()
    }
    monkeypatch.setattr(Recognizer, "_available_map", staticmethod(lambda: fake))


class TestConstants:
    """默认值与偏好序。"""

    def test_default_engine_is_the_auto_strategy(self):
        """默认是**策略名**。写死任何单个引擎都会在另一套数据上吃亏。"""
        assert EngineConfig().default == "auto"

    def test_auto_is_a_legal_engine_choice(self):
        """策略名也得在 ``ENGINE_CHOICES`` 里，否则 config.validate 会拒。"""
        assert "auto" in ENGINE_CHOICES

    def test_best_order_is_ranked_by_measured_accuracy(self):
        """顺序不是拍脑袋：100 张 CCPD 上 89% > 79% > 0%。"""
        assert BEST_ENGINE_ORDER == ("yolo", "deep", "cv")

    def test_best_order_is_not_the_cascade_order(self):
        """两个策略序必须各是各的：合并会让 ``cascade`` 悄悄变成 yolo。

        ``cascade`` 的语义是"deep 的别名"（用户明确选的），``auto`` 才是
        "你替我挑最强的"。把它们并成一份顺序，等于让 ``cascade`` 名不副实。
        """
        assert BEST_ENGINE_ORDER != CASCADE_RESOLVE_ORDER
        assert BEST_ENGINE_ORDER[0] not in CASCADE_RESOLVE_ORDER

    def test_alias_table_points_at_the_shared_constant(self):
        """别名表引用 ``config`` 的同一常量，而不是自己抄一份顺序。"""
        assert ALIAS_ENGINES["auto"]["resolve_order"] == BEST_ENGINE_ORDER


class TestResolve:
    """``Recognizer._resolve_engine`` 的解析与退化。"""

    def test_picks_yolo_when_weights_are_present(self, monkeypatch):
        _fake_availability(monkeypatch, {"yolo": True, "deep": True, "cv": True})
        assert Recognizer._resolve_engine("auto") == "yolo"

    def test_falls_back_to_deep_without_yolo_weights(self, monkeypatch):
        """没装权重 = 与旧默认（cascade）完全一致，不是新故障。"""
        _fake_availability(monkeypatch, {"yolo": False, "deep": True, "cv": True})
        assert Recognizer._resolve_engine("auto") == "deep"

    def test_falls_back_to_cv_when_deep_is_missing(self, monkeypatch):
        _fake_availability(monkeypatch, {"yolo": False, "deep": False, "cv": True})
        assert Recognizer._resolve_engine("auto") == "cv"

    def test_raises_when_nothing_is_available(self, monkeypatch):
        _fake_availability(monkeypatch, {"yolo": False, "deep": False, "cv": False})
        with pytest.raises(EngineUnavailable):
            Recognizer._resolve_engine("auto")

    def test_cascade_still_resolves_to_deep_even_with_yolo_available(self, monkeypatch):
        """``cascade`` 的语义不受 ``auto`` 影响 —— 它仍是 deep 的别名。"""
        _fake_availability(monkeypatch, {"yolo": True, "deep": True, "cv": True})
        assert Recognizer._resolve_engine("cascade") == "deep"

    @pytest.mark.parametrize("name", ["yolo", "deep", "cv", "fusion"])
    def test_explicit_engine_names_pass_through_untouched(self, monkeypatch, name):
        """用户点名哪个就是哪个 —— 不替他"优化"成更强的引擎。"""
        _fake_availability(monkeypatch, {"yolo": True, "deep": True, "cv": True, "fusion": True})
        assert Recognizer._resolve_engine(name) == name


class TestCatalog:
    """引擎清单里的 ``auto``。"""

    def _row(self) -> dict:
        return next(r for r in describe_engine_choices() if r["name"] == "auto")

    def test_auto_is_listed_first(self):
        """默认项排在最前。排在末尾会让人以为它是个冷门选项。"""
        assert describe_engine_choices()[0]["name"] == "auto"

    def test_auto_is_flagged_as_strategy(self):
        assert self._row()["strategy"] is True

    def test_auto_available_mirrors_yolo(self):
        row = self._row()
        assert row["available"] is True
        # 说明里点明"有权重就用 yolo"，否则用户不知道它到底跑谁
        assert "yolo" in row["description"]

    def test_auto_degrades_to_deep_and_says_so(self, monkeypatch):
        """yolo 不可用：仍可用，但说明必须写明退化到谁。"""
        from lpr.engines import YoloEngine

        monkeypatch.setattr(
            YoloEngine, "is_available", classmethod(lambda _cls: (False, "没有权重"))
        )
        row = self._row()
        assert row["available"] is True
        assert "退化为 DEEP" in row["description"].upper()

    def test_auto_unavailable_when_nothing_to_resolve(self, monkeypatch):
        from lpr.engines import CVEngine, DeepEngine, YoloEngine

        for cls in (YoloEngine, DeepEngine, CVEngine):
            monkeypatch.setattr(
                cls, "is_available", classmethod(lambda _cls: (False, "测试里置为不可用"))
            )
        row = self._row()
        assert row["available"] is False
        for name in BEST_ENGINE_ORDER:
            assert name in row["reason"]


class TestFusionClearFirst:
    """融合 ``auto`` 策略的首选成员也跟着走 ``auto``。"""

    def test_default_clear_first_is_auto(self):
        assert EngineConfig().fusion_auto_clear_first == "auto"
        EngineConfig().validate()  # 不抛

    @pytest.mark.parametrize("name", ["yolo", "deep", "cv", "auto"])
    def test_accepts_members_and_auto(self, name):
        EngineConfig(fusion_auto_clear_first=name).validate()

    @pytest.mark.parametrize("name", ["bogus", "fusion", "cascade"])
    def test_rejects_non_members(self, name):
        """``cascade`` / ``fusion`` 都不能当成员：一个是别名，一个是自己。"""
        with pytest.raises(ValueError, match="fusion_auto_clear_first"):
            EngineConfig(fusion_auto_clear_first=name).validate()


# --------------------------------------------------------------------- 界面


@pytest.fixture
def config_path(monkeypatch, tmp_path):
    """把 ``Config.save()`` 的目标挪到临时目录，别碰仓库的 config.json。"""
    import lpr.config as config_mod

    target = tmp_path / "config.json"
    monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_PATH", target)
    return target


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class TestGuiDefault:
    """界面下拉框必须能选中 ``auto``，且不改写配置。"""

    def test_default_auto_is_selected_and_not_rewritten(self, qt_app, config_path, caplog):
        from lpr.ui.window import MainWindow

        cfg = Config()  # 不指定引擎 = 用新默认 "auto"
        assert cfg.engine.default == "auto"
        win = MainWindow(cfg)
        try:
            with caplog.at_level(logging.WARNING, logger="lpr.ui.window"):
                win._refresh_engine_list()
            assert win.engine_combo.currentData() == "auto"
            assert win.config.engine.default == "auto"
            # 不能重演 v2.7.4："配置的默认引擎 auto 不可用" 这类假警报
            assert "不可用" not in caplog.text
        finally:
            win.close()
