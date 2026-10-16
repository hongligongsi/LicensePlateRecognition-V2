"""引擎清单契约：``cascade`` 这类**策略别名**必须能被选中，而不是被当成不可用。

回归背景（v2.7.4）——界面日志里出现::

    配置的默认引擎 cascade 不可用，界面实际使用 cv

这句话有两处错：``cascade`` 不是"不可用"，它是 ``deep`` 的别名；
而且界面**静默**把 ``config.engine.default`` 改写成了 ``cv``。
真实代价（``tests/baselines/ground_truth.json`` 的 11 张人工真值样张）::

    engine=cv       命中 0/11     ← 界面降级后跑的
    engine=cascade  命中 10/11    ← 配置里写的（解析成 deep）

根因：下拉框只列 ``describe_engines()`` 的**真引擎**，而 ``cascade``
（``config.ENGINE_CHOICES`` 里的合法取值）不在其中，于是匹配不到默认项、
回落到第一项。这套测试锁住三件事：

1. 别名仍**不**进注册表（``get_engine("cascade")`` 依然查无此引擎）；
2. ``describe_engine_choices()`` 把别名排在它实际使用的引擎之后，可用性跟随
   该引擎、退化时说明写在 ``description`` 里；
3. 界面与 CLI 都能选中它，且**不再改写** ``config.engine.default``。
"""

from __future__ import annotations

import argparse
import json
import logging

import pytest

from lpr.api import Recognizer
from lpr.config import CASCADE_RESOLVE_ORDER, Config
from lpr.engines import (
    ALIAS_ENGINES,
    CVEngine,
    DeepEngine,
    available_engines,
    describe_engine_choices,
    describe_engines,
    get_engine,
)


#: 只保留真引擎（``strategy`` 标记为假或缺失）
def _real_names() -> list[str]:
    return [row["name"] for row in describe_engines()]


def _make_unavailable(monkeypatch, cls, reason: str = "测试里故意置为不可用") -> None:
    monkeypatch.setattr(cls, "is_available", classmethod(lambda _cls: (False, reason)))


class TestAliasCatalog:
    """别名目录本身的行为。"""

    def test_cascade_is_still_not_a_registered_engine(self):
        """别名**不能**注册成引擎类。

        它一旦进 ``_REGISTRY``：``available_engines()`` 会多出一个假的实现名，
        ``FusionEngine`` 的成员过滤（``NON_MEMBER_NAMES`` 里含 ``cascade``）
        也会随之失效 —— 融合会把"别名"当成一个可投票的成员。
        """
        assert "cascade" not in available_engines()
        with pytest.raises(KeyError):
            get_engine("cascade")

    def test_choices_are_a_superset_of_real_engines(self):
        """真引擎一个都不能少，顺序也不动（下拉框顺序 = 引擎清单顺序）。"""
        assert _real_names() == [
            row["name"] for row in describe_engine_choices() if not row.get("strategy")
        ]

    def test_alias_sits_right_after_the_engine_it_uses(self):
        """``cascade`` 紧跟 ``deep``，而不是按字母序落到最后。"""
        names = [row["name"] for row in describe_engine_choices()]
        assert names.index("cascade") == names.index("deep") + 1

    def test_alias_row_is_flagged_as_strategy(self):
        row = next(r for r in describe_engine_choices() if r["name"] == "cascade")
        assert row["strategy"] is True

    def test_alias_is_available_when_deep_is(self):
        """正常情况下 ``cascade`` 可用，说明里点明它等同 deep。"""
        row = next(r for r in describe_engine_choices() if r["name"] == "cascade")
        assert row["available"] is True
        assert "deep" in row["description"]
        assert row["reason"] == ""

    def test_alias_degrades_to_cv_and_says_so(self, monkeypatch):
        """deep 不可用但 cv 可用：别名仍可用，但说明必须写明退化。

        说明字段就是界面提示语（``_update_engine_hint`` 读 ``description``），
        不写明退化就会让人以为跑的仍是 deep。
        """
        _make_unavailable(monkeypatch, DeepEngine)
        row = next(r for r in describe_engine_choices() if r["name"] == "cascade")
        assert row["available"] is True
        assert "退化为 CV" in row["description"]

    def test_alias_unavailable_when_nothing_to_resolve(self, monkeypatch):
        """deep 与 cv 都不可用：别名判为不可用，原因里两个都点名。"""
        _make_unavailable(monkeypatch, DeepEngine, "缺 onnxruntime")
        _make_unavailable(monkeypatch, CVEngine, "缺 opencv-python")
        row = next(r for r in describe_engine_choices() if r["name"] == "cascade")
        assert row["available"] is False
        assert "deep" in row["reason"] and "cv" in row["reason"]

    def test_resolve_order_is_the_one_recognizer_uses(self, monkeypatch):
        """界面算可用性用的顺序，必须与 ``Recognizer`` 真解析的一致。

        两边分叉的后果：界面说"可用"、运行时却解析成 cv（或反过来）。
        """
        assert Recognizer._resolve_engine("cascade") == CASCADE_RESOLVE_ORDER[0]
        _make_unavailable(monkeypatch, DeepEngine)
        assert Recognizer._resolve_engine("cascade") == CASCADE_RESOLVE_ORDER[1]

    def test_alias_table_points_at_shared_constant(self):
        """别名表引用 ``config`` 的同一常量，而不是自己抄一份顺序。"""
        assert ALIAS_ENGINES["cascade"]["resolve_order"] == CASCADE_RESOLVE_ORDER


@pytest.fixture
def config_path(monkeypatch, tmp_path):
    """把 ``Config.save()`` 的目标挪到临时目录，别碰仓库的 config.json。

    v2.7.4 起界面里"用户主动切换引擎/主题"会落盘，所以**每个**建窗口的用例
    都必须隔离保存路径 —— 否则一次 ``setCurrentIndex`` 就把仓库的
    ``config.json`` 重写了（写测试时真踩到过：引擎被改成 deep、文件里多出
    40 行默认值）。
    """
    import lpr.config as config_mod

    target = tmp_path / "config.json"
    monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_PATH", target)
    return target


@pytest.fixture
def qt_app(monkeypatch):
    """建一个离屏 QApplication（环境不具备 Qt 平台库时整组跳过）。

    必须 ``importorskip`` 到**子模块**：PySide6 顶层 import 不加载 Qt 平台库，
    ``QtWidgets`` 才加载（Ubuntu CI 缺 ``libEGL.so.1`` 时抛的就是它）。
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def main_window(qt_app, config_path):
    from lpr.ui.window import MainWindow

    cfg = Config()
    # 与仓库 config.json 一致：默认引擎写的就是别名 cascade
    cfg.engine.default = "cascade"
    win = MainWindow(cfg)
    yield win
    win.close()


class TestGuiEngineCombo:
    """界面下拉框。"""

    def test_configured_default_is_selected_and_not_rewritten(self, main_window):
        """核心回归：配置写 cascade 时，下拉框选中它、且配置不被改写。"""
        combo = main_window.engine_combo
        assert combo.currentData() == "cascade"
        assert main_window.config.engine.default == "cascade"

    def test_no_bogus_warning_about_alias_being_unavailable(self, main_window, caplog):
        """不该再出现「配置的默认引擎 cascade 不可用」这类假警报。"""
        with caplog.at_level(logging.WARNING, logger="lpr.ui.window"):
            main_window._refresh_engine_list()
        assert "不可用" not in caplog.text

    def test_every_choice_is_listed_in_the_combo(self, main_window):
        combo = main_window.engine_combo
        listed = [combo.itemData(i) for i in range(combo.count())]
        assert listed == [row["name"] for row in describe_engine_choices()]

    def test_statusbar_shows_the_display_name(self, main_window):
        """状态栏给人看，应该是「级联引擎」而不是内部名 ``cascade``。"""
        assert main_window._status_engine.text() == "引擎：级联引擎"

    def test_hint_explains_what_the_alias_means(self, main_window):
        assert "deep" in main_window.engine_hint.text()

    def test_switching_engine_updates_statusbar(self, main_window):
        combo = main_window.engine_combo
        index = combo.findData("deep")
        assert index >= 0
        combo.setCurrentIndex(index)
        assert main_window.config.engine.default == "deep"
        assert main_window._status_engine.text() == "引擎：深度学习引擎"

    def test_unavailable_default_falls_back_with_a_warning(
        self, qt_app, monkeypatch, caplog, config_path
    ):
        """配置写了**真**不可用的引擎时：仍回落，但要 warning 且说清原因。

        回落本身是必要的（否则界面起不来），但不能是静默的 —— 这是用户
        能感知到的功能变化，而且配置文件里的值并没有被改写。
        """
        from lpr.ui.window import MainWindow

        cfg = Config()
        win = MainWindow(cfg)  # 先按正常环境建起来
        try:
            _make_unavailable(monkeypatch, CVEngine, "缺 opencv-python")
            # 构造之后再指向一个真不可用的引擎，才能观察"回落"那一步
            win.config.engine.default = "cv"
            with caplog.at_level(logging.WARNING, logger="lpr.ui.window"):
                win._refresh_engine_list()

            assert "本次运行回落到" in caplog.text
            assert "config.json 未改动" in caplog.text
            assert "缺 opencv-python" in caplog.text  # 原因要带上，否则用户无从下手
            assert win.config.engine.default == win.engine_combo.currentData()
            assert win.config.engine.default != "cv"
        finally:
            win.close()

    def test_fallback_never_lands_on_a_disabled_item(self, qt_app, monkeypatch, config_path):
        """回落目标必须是**可用**项 —— 第 0 项自己也可能不可用。

        回归：``default_idx`` 原先是硬编码的 0，而 ``setCurrentIndex`` 并不理会
        单项的 enabled 状态，于是"回落"会把一个置灰的引擎选成当前项：界面看着
        选好了，实际跑不了。
        """
        from lpr.ui.window import MainWindow

        _make_unavailable(monkeypatch, CVEngine, "缺 opencv-python")
        cfg = Config()
        cfg.engine.default = "cv"  # 故意指向不可用的那个
        win = MainWindow(cfg)
        try:
            combo = win.engine_combo
            assert combo.count() > 0
            model = combo.model()
            assert model.item(0).isEnabled() is False  # 第 0 项确实被置灰了
            chosen = combo.currentData()
            assert chosen != "cv"
            index = combo.findData(chosen)
            assert model.item(index).isEnabled() is True
            # 状态栏与它一致（这条曾经不一致过：显示 cascade、实跑 cv）
            assert win._status_engine.text().endswith(
                next(r["display_name"] for r in describe_engine_choices() if r["name"] == chosen)
            )
        finally:
            win.close()


class TestCliEngines:
    """``lpr engines`` 表格。"""

    def test_lists_alias_with_footnote_and_configured_default(self, capsys):
        from lpr.cli import cmd_engines

        cfg = Config()
        assert cmd_engines(argparse.Namespace(), cfg) == 0
        out = capsys.readouterr().out
        assert "级联引擎*" in out
        assert "策略名" in out  # 脚注解释星号的含义
        assert f"engine.default = {cfg.engine.default}" in out
        assert "可用" in out

    def test_says_when_the_configured_default_is_unknown(self, capsys):
        """配置里写了个清单外的名字：如实说会回落，而不是假装可用。"""
        from lpr.cli import cmd_engines

        cfg = Config()
        cfg.engine.default = "not-an-engine"
        cmd_engines(argparse.Namespace(), cfg)
        out = capsys.readouterr().out
        assert "不在清单里" in out


class TestConfigPersistence:
    """界面里「用户主动改的偏好」落盘，「自动回落」绝不落盘。

    这条区分是 v2.7.4 的核心：原先配置里的 ``cascade`` 被自动改写成 ``cv``，
    一旦落盘就是**永久**降级（准确率 10/11 → 0/11）。自动回落只能改内存，
    用户主动选的才写回文件。
    """

    def test_theme_choice_is_written_back(self, main_window, config_path):
        """切主题要真的写回文件（此前只改内存，重开就回到浅色）。"""
        main_window.on_theme_changed("dark")

        assert config_path.exists()
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        assert saved["ui"]["theme"] == "dark"

    def test_engine_choice_is_written_back(self, main_window, config_path):
        combo = main_window.engine_combo
        index = combo.findData("deep")
        assert index >= 0
        combo.setCurrentIndex(index)

        saved = json.loads(config_path.read_text(encoding="utf-8"))
        assert saved["engine"]["default"] == "deep"

    def test_automatic_fallback_does_not_touch_the_file(
        self, qt_app, monkeypatch, config_path
    ):
        """配置里的引擎不可用时只回落内存，**不改**用户的配置文件。"""
        from lpr.ui.window import MainWindow

        _make_unavailable(monkeypatch, CVEngine, "缺 opencv-python")
        cfg = Config()
        cfg.engine.default = "cv"
        win = MainWindow(cfg)
        try:
            assert win.config.engine.default != "cv"  # 确实回落了
            assert not config_path.exists()  # 但一个字都没写进文件
        finally:
            win.close()
