"""运行环境指纹：一行日志说清"这次跑的是什么"（v2.8.1）。

排查"同一张图昨天能认、今天认不出"时，旧日志里只有引擎就绪那行，分不清是
**代码变了 / 配置变了 / 环境变了** —— 三者修法完全不同。这套测试锁住：

1. 行里必须同时有版本、Python、引擎默认值、配置指纹（缺一项就答不了那个问题）；
2. 指纹只对**影响识别结果**的配置敏感：改阈值要变，改主题/日志级别不能变；
3. 没加载的重量级依赖（cv2/numpy）如实写"未加载"，不为了打印版本号去 import。
"""

from __future__ import annotations

import re

from lpr import __version__
from lpr.config import Config
from lpr.utils.environment import (
    _module_version,
    config_fingerprint,
    environment_line,
    log_environment,
)


class TestLine:
    def test_contains_everything_needed_to_reproduce(self):
        line = environment_line(Config())
        assert f"lpr {__version__}" in line
        assert "Python " in line
        assert f"引擎默认={Config().engine.default}" in line
        assert "配置指纹=" in line
        assert "\n" not in line  # 单行：日志按行 grep 才方便

    def test_fingerprint_is_short_hex(self):
        fp = config_fingerprint(Config())
        assert re.fullmatch(r"[0-9a-f]{8}", fp), fp

    def test_is_logged_at_info(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="lpr.utils.environment"):
            returned = log_environment(Config())
        assert "环境指纹" in caplog.text
        # 返回值与日志内容一致（不重复格式化两份）
        assert returned in caplog.text


class TestFingerprintSensitivity:
    """指纹的作用是回答"两次是不是同一份配置"，所以要**该变的变、不该变的不变**。"""

    def test_same_config_gives_same_fingerprint(self):
        assert config_fingerprint(Config()) == config_fingerprint(Config())

    def test_threshold_change_moves_the_fingerprint(self):
        cfg = Config()
        before = config_fingerprint(cfg)
        cfg.engine.locate.open_kernel_ratio = cfg.engine.locate.open_kernel_ratio + 0.01
        assert config_fingerprint(cfg) != before

    def test_engine_choice_moves_the_fingerprint(self):
        cfg = Config()
        before = config_fingerprint(cfg)
        cfg.engine.default = "deep"
        assert config_fingerprint(cfg) != before

    def test_fusion_threshold_moves_the_fingerprint(self):
        cfg = Config()
        before = config_fingerprint(cfg)
        cfg.engine.fusion_min_support = 1
        assert config_fingerprint(cfg) != before

    def test_ui_and_logging_do_not_move_the_fingerprint(self):
        """改主题/日志级别不会让同一张图读出不同的车牌，混进来只是噪音。"""
        cfg = Config()
        before = config_fingerprint(cfg)
        cfg.ui.theme = "dark"
        cfg.log_level = "DEBUG"
        assert config_fingerprint(cfg) == before


class TestLazyVersions:
    def test_unloaded_module_is_reported_as_such(self):
        """不替它 import —— CLI 的 config/check 子命令根本用不到 OpenCV。"""
        assert _module_version("module_that_does_not_exist") == "未加载"
