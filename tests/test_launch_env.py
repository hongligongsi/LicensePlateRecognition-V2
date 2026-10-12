"""启动入口与运行环境兼容性的回归测试。

覆盖三件容易"改着改着就退化"的事：

1. ``python -m lpr`` 入口必须存在且能把参数透传给 CLI；
2. 摄像头后端不能又变回无条件的 Windows 专有 ``CAP_DSHOW``；
3. ``start.bat`` 必须是纯 ASCII —— 批处理按系统 ANSI 代码页解析，
   文件里出现非 ASCII 会把中文拆成乱码甚至**当命令执行**（实测踩过）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    """在子进程里以 ``python -m lpr`` 方式运行（验证真实入口，而非直接调函数）。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, "-m", "lpr", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(ROOT),
        timeout=timeout,
    )


class TestModuleEntry:
    """``python -m lpr`` —— 不装 console script 也要能跑。"""

    def test_main_module_exists(self):
        assert (SRC / "lpr" / "__main__.py").is_file()

    def test_bare_invocation_defaults_to_gui(self):
        src = (SRC / "lpr" / "__main__.py").read_text(encoding="utf-8")
        # 无参数 -> 注入 gui 子命令（而不是让 argparse 报"缺少子命令"）
        assert '"gui"' in src
        assert "sys.exit(main(" in src

    def test_arguments_are_forwarded_to_cli(self):
        proc = _run(["engines"])
        assert proc.returncode == 0, proc.stderr
        assert "传统 CV 引擎" in proc.stdout

    def test_unknown_argument_still_errors_like_cli(self):
        proc = _run(["--definitely-not-a-command"])
        assert proc.returncode != 0


class TestCameraBackend:
    """摄像头后端必须按平台选择（``CAP_DSHOW`` 是 Windows 专有）。"""

    def test_backend_follows_platform(self, monkeypatch):
        import cv2

        from lpr.ui import worker

        monkeypatch.setattr(worker.sys, "platform", "win32")
        assert worker.camera_backend() == cv2.CAP_DSHOW

        monkeypatch.setattr(worker.sys, "platform", "linux")
        assert worker.camera_backend() == cv2.CAP_ANY

        monkeypatch.setattr(worker.sys, "platform", "darwin")
        assert worker.camera_backend() == cv2.CAP_ANY

    def test_source_has_no_unconditional_dshow(self):
        src = (SRC / "lpr" / "ui" / "worker.py").read_text(encoding="utf-8")
        # 旧的坏写法是 VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        # 直接写在采集循环里（docstring 里作为反例提到该写法是允许的）
        assert "VideoCapture(self.camera_index, cv2.CAP_DSHOW)" not in src
        # 新写法必须走平台判断
        assert "cv2.VideoCapture(self.camera_index, camera_backend())" in src


class TestWindowsLauncher:
    """``start.bat`` 必须能被 cmd 正确解析。"""

    def test_launcher_exists_and_is_ascii_only(self):
        bat = ROOT / "start.bat"
        assert bat.is_file()
        # 关键护栏：非 ASCII 字节会让 cmd 按 GBK 解析出错行（实测）
        bat.read_bytes().decode("ascii")

    def test_launcher_sets_pythonpath_and_calls_module_entry(self):
        text = (ROOT / "start.bat").read_text(encoding="ascii")
        assert "PYTHONPATH=%~dp0src" in text
        assert "%PY% -m lpr" in text
        # 无参数时默认开界面
        assert 'set "ARGS=gui"' in text
        # 版本过低要明确报错，而不是抛一堆 ImportError
        assert "3, 10" in text

    def test_launcher_does_not_use_crlf_hostile_constructs(self):
        """CRLF 是批处理必需的；这里确认没被误转成 LF。"""
        raw = (ROOT / "start.bat").read_bytes()
        assert b"\r\n" in raw


@pytest.mark.parametrize("doc", ["README.md", "docs/UPGRADE.md"])
def test_docs_mention_python_requirement(doc: str):
    """系统/环境要求必须写在文档里 —— 这是用户最常问的第一个问题。"""
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert "3.10" in text
