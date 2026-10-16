"""运行日志的测试。

三组关注点，对应三个真实出问题的地方：

1. **落盘**（``TestFileSink``）：文件建在哪、叫什么名字、写进去的是什么、
   超过上限怎么轮转、目录不可写时会不会把程序带崩；
2. **幂等与重建**（``TestHandlerLifecycle``）：CLI / 界面 / 批量子进程三处
   入口都会调 ``setup_logging``，重复调用既不能叠加 handler，也不能把先配好的
   级别悄悄改回去；
3. **接线**（``TestConfigWiring`` / ``TestCommandLine`` / ``TestMemoryHandler``）：
   ``Config.logging_kwargs`` 与 ``setup_logging`` 的形参、``--log-*`` 开关、
   以及"新出口必须同样脱敏"这条隐私约束。

界面部分（``TestGuiLogPanel``）钉住一个真 bug：v2.7.3 之前「运行日志」面板
一旦攒够 20 行就再也不刷新。
"""

from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from lpr.cli import _apply_log_cli, _make_parser
from lpr.cli import main as cli_main
from lpr.config import Config
from lpr.utils.logging_setup import (
    DEFAULT_LOG_DIR,
    LEVEL_CHOICES,
    RingBufferHandler,
    _coerce_level,
    close_logging,
    current_log_file,
    log_file_path,
    memory_handler,
    setup_logging,
)

#: ``SecurityContext.setup()`` 会挂脱敏过滤器的 logger；测试里要还原，
#: 否则一条 ``皖A12345`` 就会污染后续所有用例的日志断言
_MASKED_LOGGERS = ("lpr", "lpr.engines", "lpr.engines.cv", "lpr.engines.deep", "lpr.ui")


def _installed(logger: logging.Logger) -> list[logging.Handler]:
    """只数**本模块**装上的 handler。

    跑测试时 ``lpr`` 这个 logger 上还会挂着一两个 ``LogCaptureHandler``
    （pytest 的 logging 插件，或本机测试环境自己的捕获器）。按 ``len(handlers)``
    断言会随环境随机失败，所以这里按类型认领：stderr 流（``type(...) is``，
    刻意不用 ``isinstance`` —— ``LogCaptureHandler`` 也是 ``StreamHandler``
    的子类）、内存缓冲、以及文件出口。
    """
    ours: list[logging.Handler] = []
    for handler in logger.handlers:
        if (
            handler is memory_handler
            or isinstance(handler, RotatingFileHandler)
            or type(handler) is logging.StreamHandler
        ):
            ours.append(handler)
    return ours


@pytest.fixture(autouse=True)
def clean_logging():
    """每个用例前后都把全局日志状态复原（它是模块级单例，不隔离会互相干扰）。"""
    saved = {name: list(logging.getLogger(name).filters) for name in _MASKED_LOGGERS}
    close_logging()
    memory_handler.clear()
    yield
    close_logging()
    memory_handler.clear()
    for name, filters in saved.items():
        logging.getLogger(name).filters = filters
    logging.getLogger("lpr").setLevel(logging.NOTSET)


# ══════════════════════════════════════════════════════════════════
# 一、路径与文件名
# ══════════════════════════════════════════════════════════════════


class TestLogFilePath:
    def test_default_name_carries_the_date(self, tmp_path):
        path = log_file_path(tmp_path, when=datetime(2026, 1, 2, 3, 4, 5))
        assert path == tmp_path / "lpr-20260102.log"

    def test_tag_isolates_child_processes(self, tmp_path):
        """多进程各写各的：同一个文件被两个进程轮流轮转会在 Windows 上抛异常。"""
        path = log_file_path(tmp_path, when=datetime(2026, 1, 2), tag="-p4321")
        assert path.name == "lpr-20260102-p4321.log"

    def test_relative_dir_is_kept_relative(self):
        """相对目录不能被悄悄展开成绝对路径：日志该跟着当前工作目录走。"""
        assert log_file_path(DEFAULT_LOG_DIR, when=datetime(2026, 1, 2)) == Path(
            "logs/lpr-20260102.log"
        )

    def test_default_dir_matches_config_default(self):
        """``Config.log_dir`` 与 ``DEFAULT_LOG_DIR`` 必须一致（否则文档与代码各说各话）。"""
        assert Config().log_dir == DEFAULT_LOG_DIR


# ══════════════════════════════════════════════════════════════════
# 二、级别解析
# ══════════════════════════════════════════════════════════════════


class TestLevelParsing:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [("DEBUG", 10), ("info", 20), ("Warning", 30), ("error", 40), ("CRITICAL", 50)],
    )
    def test_names_are_case_insensitive(self, given, expected):
        assert _coerce_level(given)[0] == expected

    @pytest.mark.parametrize(("alias", "real"), [("WARN", 30), ("FATAL", 50), ("warn", 30)])
    def test_logging_aliases_are_accepted(self, alias, real):
        assert _coerce_level(alias) == (real, False)

    def test_integer_passes_through(self):
        assert _coerce_level(logging.ERROR) == (logging.ERROR, False)

    def test_unknown_name_falls_back_to_info_and_is_flagged(self):
        level, unknown = _coerce_level("TRACE")
        assert (level, unknown) == (logging.INFO, True)

    def test_choices_are_all_real_logging_levels(self):
        """防 ``getattr(logging, name)`` 撞上 ``raiseExceptions`` 这类整型常量。"""
        for name in LEVEL_CHOICES:
            assert isinstance(getattr(logging, name), int)
        assert "raiseExceptions" not in LEVEL_CHOICES

    def test_unknown_level_is_reported_not_swallowed(self, tmp_path, capsys):
        setup_logging("TRACE", log_dir=tmp_path, console=True)
        assert "未知日志级别" in capsys.readouterr().err


# ══════════════════════════════════════════════════════════════════
# 三、文件出口
# ══════════════════════════════════════════════════════════════════


class TestFileSink:
    def test_creates_dir_and_file(self, tmp_path):
        setup_logging(log_dir=tmp_path / "深层" / "logs")
        path = current_log_file()
        assert path is not None and path.exists()
        assert path.parent == tmp_path / "深层" / "logs"

    def test_records_land_in_the_file(self, tmp_path):
        setup_logging(log_dir=tmp_path, console=False)
        logging.getLogger("lpr").info("识别完成 %s", "测试消息")
        text = current_log_file().read_text(encoding="utf-8")
        assert "识别完成 测试消息" in text
        assert "[INFO]" in text

    def test_first_line_tells_where_the_log_is(self, tmp_path):
        """用户最高频的追问是"日志在哪"，第一行就必须回答它。"""
        setup_logging(log_dir=tmp_path, console=False)
        first = current_log_file().read_text(encoding="utf-8").splitlines()[0]
        assert "运行日志 →" in first
        assert str(tmp_path.absolute()) in first

    def test_chinese_is_written_as_utf8(self, tmp_path):
        setup_logging(log_dir=tmp_path, console=False)
        logging.getLogger("lpr").info("皖A·新能源")
        assert "皖A·新能源" in current_log_file().read_text(encoding="utf-8")

    def test_disabled_writes_nothing(self, tmp_path):
        setup_logging(log_to_file=False, log_dir=tmp_path / "logs")
        assert current_log_file() is None
        assert not (tmp_path / "logs").exists()

    def test_empty_dir_string_also_means_disabled(self, tmp_path):
        setup_logging(log_dir="", log_to_file=True)
        assert current_log_file() is None

    def test_unwritable_dir_degrades_to_console_only(self, tmp_path, capsys):
        """写不了日志不能让程序起不来 —— 真正需要日志的时刻恰是环境不正常时。"""
        blocker = tmp_path / "blocker"
        blocker.write_text("我是文件，不是目录", encoding="utf-8")

        setup_logging(log_dir=blocker / "logs", console=True)

        assert current_log_file() is None
        assert "无法写入日志文件" in capsys.readouterr().err
        logging.getLogger("lpr").info("仍然可以继续跑")
        assert "仍然可以继续跑" in capsys.readouterr().err

    def test_rolls_over_when_over_the_size_limit(self, tmp_path):
        setup_logging(log_dir=tmp_path, max_mb=1, backups=2, console=False)
        log = logging.getLogger("lpr")
        blob = "x" * 600_000
        for index in range(4):
            log.info("第 %d 条 %s", index, blob)

        base = current_log_file()
        assert base is not None
        assert base.with_name(base.name + ".1").exists(), "超过 max_mb 必须轮转出 .1"
        assert base.stat().st_size < 1_100_000

    def test_zero_backups_keeps_no_history(self, tmp_path):
        setup_logging(log_dir=tmp_path, max_mb=1, backups=0, console=False)
        log = logging.getLogger("lpr")
        for index in range(4):
            log.info("第 %d 条 %s", index, "y" * 600_000)

        base = current_log_file()
        assert base is not None
        assert not base.with_name(base.name + ".1").exists()


# ══════════════════════════════════════════════════════════════════
# 四、handler 生命周期
# ══════════════════════════════════════════════════════════════════


class TestHandlerLifecycle:
    def test_second_call_does_not_stack_handlers(self, tmp_path):
        logger = setup_logging(log_dir=tmp_path)
        before = list(logger.handlers)
        again = setup_logging(log_dir=tmp_path)
        assert again is logger
        assert list(again.handlers) == before

    def test_second_call_does_not_reset_the_level(self, tmp_path):
        """``main()`` 用 ``--log-level DEBUG`` 配好之后，子命令里的第二次调用
        不许把它改回配置文件里的 INFO。"""
        logger = setup_logging("DEBUG", log_dir=tmp_path)
        setup_logging("WARNING", log_dir=tmp_path)
        assert logger.level == logging.DEBUG

    def test_force_rebuilds_and_releases_the_old_file(self, tmp_path):
        setup_logging("INFO", log_dir=tmp_path)
        before = current_log_file()
        assert before is not None

        other = tmp_path / "other"
        logger = setup_logging("DEBUG", log_dir=other, force=True)

        assert logger.level == logging.DEBUG
        assert len(_installed(logger)) == 3  # stderr + 内存缓冲 + 文件
        assert current_log_file() is not None and current_log_file().parent == other
        # 旧句柄没关掉的话，Windows 上这一行会抛 PermissionError
        before.unlink()

    def test_memory_handler_is_attached_exactly_once(self, tmp_path):
        logger = setup_logging("INFO", log_dir=tmp_path)
        assert logger.handlers.count(memory_handler) == 1
        setup_logging("INFO", log_dir=tmp_path, force=True)
        assert logger.handlers.count(memory_handler) == 1

    def test_console_can_be_turned_off(self, tmp_path):
        logger = setup_logging(log_dir=tmp_path, console=False)
        assert not any(type(h) is logging.StreamHandler for h in logger.handlers)

    def test_propagate_is_off(self, tmp_path):
        """开着 propagate 会让日志顺着根 logger 走：INFO 被丢、WARNING 重复。"""
        logger = setup_logging(log_dir=tmp_path)
        assert logger.propagate is False

    def test_close_detaches_everything(self, tmp_path):
        logger = setup_logging(log_dir=tmp_path)
        assert _installed(logger)
        close_logging()
        assert _installed(logger) == []
        assert current_log_file() is None
        # 关掉之后必须能重新配置（否则"关一次就再也开不起来"）
        setup_logging(log_dir=tmp_path)
        assert current_log_file() is not None


# ══════════════════════════════════════════════════════════════════
# 五、内存缓冲（界面「运行日志」面板的数据源）
# ══════════════════════════════════════════════════════════════════


class TestMemoryHandler:
    def test_ring_buffer_keeps_the_last_n(self):
        handler = RingBufferHandler(capacity=3)
        handler.setFormatter(logging.Formatter("%(message)s"))
        for index in range(5):
            handler.emit(logging.LogRecord("t", logging.INFO, "f", 1, f"第 {index} 条", (), None))
        assert handler.dump() == ["第 2 条", "第 3 条", "第 4 条"]

    def test_dump_is_a_copy(self):
        handler = RingBufferHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.emit(logging.LogRecord("t", logging.INFO, "f", 1, "一条", (), None))
        snapshot = handler.dump()
        snapshot.append("外部改动")
        assert handler.dump() == ["一条"]

    def test_setup_fills_the_shared_buffer(self, tmp_path):
        setup_logging(log_dir=tmp_path, console=False)
        logging.getLogger("lpr").info("面板要能看到这一条")
        assert any("面板要能看到这一条" in line for line in memory_handler.dump())

    def test_same_buffer_instance_is_shared(self, tmp_path):
        """界面读的是模块级单例，不能每次 setup 换一个新对象。"""
        setup_logging(log_dir=tmp_path)
        logging.getLogger("lpr").info("共享缓冲")
        setup_logging(log_dir=tmp_path, force=True)
        assert any("共享缓冲" in line for line in memory_handler.dump())


# ══════════════════════════════════════════════════════════════════
# 六、配置与命令行接线
# ══════════════════════════════════════════════════════════════════


class TestConfigWiring:
    def test_kwargs_names_match_the_function_signature(self):
        """字段名改一个字母就会静默失联（``setup_logging`` 收到不认识的 kwarg 会抛，
        但改成"少传一个"就完全无声），所以用签名钉死。"""
        params = set(inspect.signature(setup_logging).parameters)
        assert set(Config().logging_kwargs()) <= params

    def test_kwargs_are_actually_applied(self, tmp_path):
        cfg = Config()
        cfg.log_level = "ERROR"
        cfg.log_dir = str(tmp_path / "自定义目录")
        cfg.log_to_file = True

        setup_logging(**cfg.logging_kwargs())

        assert logging.getLogger("lpr").level == logging.ERROR
        assert current_log_file() is not None
        assert current_log_file().parent == tmp_path / "自定义目录"
        assert current_log_file().exists()

    def test_tag_is_passed_through(self, tmp_path):
        cfg = Config()
        cfg.log_dir = str(tmp_path)
        setup_logging(**cfg.logging_kwargs(tag="-p999"))
        assert current_log_file().name.endswith("-p999.log")

    def test_payload_roundtrip_keeps_log_fields(self):
        cfg = Config()
        cfg.log_level = "DEBUG"
        cfg.log_dir = "自定义"
        cfg.log_to_file = False
        cfg.log_max_mb = 9
        cfg.log_backups = 1

        back = Config.from_payload(cfg.to_payload())

        assert (back.log_level, back.log_dir, back.log_to_file) == ("DEBUG", "自定义", False)
        assert (back.log_max_mb, back.log_backups) == (9, 1)

    def test_config_file_is_read(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "log_level": "WARNING",
                    "log_to_file": False,
                    "log_dir": "mylogs",
                    "log_max_mb": 2,
                    "log_backups": 3,
                }
            ),
            encoding="utf-8",
        )

        cfg = Config.load(path)

        assert cfg.log_level == "WARNING"
        assert cfg.log_to_file is False
        assert cfg.log_dir == "mylogs"
        assert (cfg.log_max_mb, cfg.log_backups) == (2, 3)

    def test_env_beats_config_file(self, tmp_path, monkeypatch):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"log_level": "WARNING", "log_dir": "from_file", "log_to_file": True}),
            encoding="utf-8",
        )
        monkeypatch.setenv("LPR_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LPR_LOG_DIR", "from_env")
        monkeypatch.setenv("LPR_LOG_FILE", "0")

        cfg = Config.load(path)

        assert (cfg.log_level, cfg.log_dir, cfg.log_to_file) == ("DEBUG", "from_env", False)

    @pytest.mark.parametrize("value", ["0", "false", "OFF", 0])
    def test_log_file_env_accepts_loose_off_values(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv("LPR_LOG_FILE", str(value))
        assert Config.load(tmp_path / "不存在.json").log_to_file is False

    def test_hand_edited_json_is_parsed_leniently(self, tmp_path):
        """config.json 是给人手改的：``"false"`` 得当假、越界值要夹住、烂值用默认。"""
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"log_to_file": "false", "log_max_mb": 0, "log_backups": "abc"}),
            encoding="utf-8",
        )

        cfg = Config.load(path)

        assert cfg.log_to_file is False
        assert cfg.log_max_mb == 1  # 下限 1MB
        assert cfg.log_backups == 5  # 读不出来 → 保持默认


class TestCommandLine:
    def test_options_work_before_and_after_the_subcommand(self):
        parser = _make_parser()
        for argv in (
            ["--log-level", "DEBUG", "predict", "a.jpg"],
            ["predict", "a.jpg", "--log-level", "DEBUG"],
            ["engines", "--log-level", "DEBUG"],
        ):
            assert parser.parse_args(argv).log_level == "DEBUG"

    def test_absent_options_are_not_set_at_all(self):
        """用 ``SUPPRESS`` 当默认值，子命令才不会用 None 覆盖主命令填好的值。"""
        args = _make_parser().parse_args(["predict", "a.jpg"])
        for name in ("log_level", "log_dir", "no_log_file"):
            assert not hasattr(args, name)

    def test_unknown_level_is_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            _make_parser().parse_args(["predict", "a.jpg", "--log-level", "TRACE"])

    def test_apply_overrides_the_config(self, tmp_path):
        cfg = Config()
        args = _make_parser().parse_args(
            [
                "predict",
                "a.jpg",
                "--log-level",
                "DEBUG",
                "--log-dir",
                str(tmp_path),
                "--no-log-file",
            ]
        )
        _apply_log_cli(cfg, args)
        assert cfg.log_level == "DEBUG"
        assert cfg.log_dir == str(tmp_path)
        assert cfg.log_to_file is False

    def test_apply_without_options_changes_nothing(self):
        cfg = Config()
        before = (cfg.log_level, cfg.log_dir, cfg.log_to_file, cfg.log_max_mb, cfg.log_backups)
        _apply_log_cli(cfg, _make_parser().parse_args(["predict", "a.jpg"]))
        assert (cfg.log_level, cfg.log_dir, cfg.log_to_file, cfg.log_max_mb, cfg.log_backups) == (
            before
        )

    def test_main_writes_a_run_log(self, tmp_path, monkeypatch, capsys):
        """主入口必须真的留下日志文件 —— 这正是用户提的"没有运行日志"。"""
        monkeypatch.chdir(tmp_path)

        assert cli_main(["engines"]) == 0

        logs = sorted((tmp_path / "logs").glob("lpr-*.log"))
        assert len(logs) == 1
        assert "运行日志 →" in logs[0].read_text(encoding="utf-8")

    def test_main_honours_no_log_file(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert cli_main(["engines", "--no-log-file"]) == 0
        assert not (tmp_path / "logs").exists()

    def test_main_honours_env_switch(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LPR_LOG_FILE", "0")
        assert cli_main(["engines"]) == 0
        assert not (tmp_path / "logs").exists()


# ══════════════════════════════════════════════════════════════════
# 七、隐私：新出口必须同样脱敏
# ══════════════════════════════════════════════════════════════════


class TestMasking:
    def test_file_sink_is_masked_too(self, tmp_path, capsys):
        """文件是新开的出口，不能成为绕过 ``mask_logs`` 的后门。"""
        from lpr.utils.security import SecurityContext

        SecurityContext(install_log_masking=True).setup()
        setup_logging(log_dir=tmp_path, console=False)

        logging.getLogger("lpr").info("识别到 %s", "皖A12345")

        text = current_log_file().read_text(encoding="utf-8")
        assert "皖A12345" not in text
        assert "*" in text


# ══════════════════════════════════════════════════════════════════
# 八、界面「运行日志」面板
# ══════════════════════════════════════════════════════════════════


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    # 这里必须 importorskip 到**子模块**，不能只写 "PySide6"：PySide6 顶层
    # import 并不加载 Qt 平台库，``QtWidgets`` 才加载。Ubuntu CI 上缺
    # ``libEGL.so.1`` 时，抛 ImportError 的正是下一行，于是 4 个界面用例会以
    # 「error / failed」的形式把 5 个 ubuntu 作业全弄红，而 Windows 上一切正常，
    # 本地根本复现不出来（v2.7.3 首次推送的真实教训）。
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def main_window(qt_app, tmp_path):
    from lpr.ui.window import MainWindow

    cfg = Config()
    cfg.log_to_file = True
    cfg.log_dir = str(tmp_path / "logs")
    setup_logging(**cfg.logging_kwargs())

    win = MainWindow(cfg)
    yield win
    win.close()


class TestGuiLogPanel:
    def test_panel_keeps_refreshing_after_it_is_full(self, main_window, tmp_path):
        """回归：旧实现用"行数没变就不再重绘"判重，攒够 20 行后面板永久冻住。"""
        log = logging.getLogger("lpr")

        for index in range(20):
            log.info("第 %d 条测试日志", index)
        main_window._refresh_log_view()
        first_rows = main_window.log_view.rowCount()
        assert first_rows > 0

        for index in range(20, 30):
            log.info("第 %d 条测试日志", index)
        main_window._refresh_log_view()

        text = "\n".join(
            main_window.log_view.item(row, 0).text()
            for row in range(main_window.log_view.rowCount())
        )
        assert "第 29 条测试日志" in text
        assert main_window.log_view.rowCount() > first_rows

    def test_panel_shows_the_log_file_path(self, main_window):
        assert "lpr-" in main_window.log_path_label.text()
        assert main_window.btn_log_dir.isEnabled()

    def test_panel_says_so_when_file_logging_is_off(self, qt_app, tmp_path):
        from lpr.ui.window import MainWindow

        close_logging()
        cfg = Config()
        cfg.log_to_file = False
        win = MainWindow(cfg)
        try:
            assert "未开启" in win.log_path_label.text()
            # 打不开的目录不给按钮，免得点了没反应像是坏了
            assert not win.btn_log_dir.isEnabled()
        finally:
            win.close()

    def test_open_directory_failure_is_reported_not_raised(self, qt_app, monkeypatch):
        """打不开文件管理器只降级为一句说明，不能把界面炸掉。

        带 ``qt_app`` 不只是为了建窗口：``lpr.ui.window`` 依赖 Qt 平台库，
        环境不具备时应当跳过而不是报错（原因见 ``qt_app`` 的注释）。
        """
        from lpr.ui import window as window_mod

        def boom(*_args, **_kwargs):
            raise OSError("模拟：没有文件管理器")

        monkeypatch.setattr(window_mod.sys, "platform", "linux")
        monkeypatch.setattr(window_mod.subprocess, "Popen", boom)

        error = window_mod._open_directory(Path("/tmp"))

        assert error is not None and "模拟" in error
