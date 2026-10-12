"""启动入口与运行环境兼容性的回归测试。

覆盖四件容易"改着改着就退化"的事：

1. ``python -m lpr`` 入口必须存在且能把参数透传给 CLI；
2. 摄像头后端不能又变回无条件的 Windows 专有 ``CAP_DSHOW``；
3. ``start.bat`` 必须是纯 ASCII —— 批处理按系统 ANSI 代码页解析，
   文件里出现非 ASCII 会把中文拆成乱码甚至**当命令执行**（实测踩过）；
4. ``start.bat`` 的依赖策略（v2.3.5 修）：探测必须覆盖核心运行库
   （numpy / OpenCV），缺失时优先建项目内 ``.venv`` —— 老版本只查
   PySide6，在干净的系统解释器上会先装 ~220 MB 再以
   ``ModuleNotFoundError: numpy`` 告终，而且把包装进了用户的全局 Python。
"""
from __future__ import annotations

import os
import re
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


class TestLauncherDependencyPolicy:
    """v2.3.5 修掉的坑：依赖探测不完整 + 往用户全局解释器里装包。

    背景（真实复现）：一台只装了官方 CPython 3.13 的机器上，
    ``import numpy`` / ``import cv2`` 都失败，而老版 start.bat 只检查
    ``import PySide6``，于是它对用户宣布"GUI dependency OK"，装了
    220 MB 的 PySide6 元包，然后在最后一步抛
    ``ModuleNotFoundError: No module named 'numpy'``——
    包还留在了用户的全局 Python 里。
    """

    @staticmethod
    def _bat() -> str:
        return (ROOT / "start.bat").read_text(encoding="ascii")

    @staticmethod
    def _probe_expression() -> str:
        m = re.search(r'^set "PROBE=(.+)"$', TestLauncherDependencyPolicy._bat(), re.M)
        assert m, "start.bat 必须定义 PROBE 表达式"
        return m.group(1)

    def test_probe_covers_core_runtime_not_only_pyside6(self):
        expr = self._probe_expression()
        # 这四项缺一不可；只写 PySide6 就是我们要防的回归
        for mod in ("numpy", "cv2", "PIL", "PySide6"):
            assert mod in expr, f"依赖探测漏了 {mod}"
        assert "PySide6" in expr and "numpy" in expr

    def test_probe_expression_is_runnable_python(self):
        """PROBE 会被拼进 ``-c`` 参数，语法必须是合法 Python 表达式。"""
        expr = self._probe_expression()
        compile(expr, "<probe>", "exec")

    def test_probe_passes_in_a_complete_environment(self):
        """在依赖齐全的解释器里，探测必须静默通过（返回码 0、无输出）。"""
        for mod in ("numpy", "cv2", "PIL", "PySide6"):
            pytest.importorskip(mod)
        proc = subprocess.run(
            [sys.executable, "-c", self._probe_expression()],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""

    def test_probe_fails_when_one_dependency_is_absent(self):
        """反向对照：少一个库就必须非零退出（否则探测形同虚设）。"""
        expr = self._probe_expression() + ", lpr_definitely_absent_module"
        proc = subprocess.run(
            [sys.executable, "-c", expr], capture_output=True, text=True, timeout=120
        )
        assert proc.returncode != 0
        assert "lpr_definitely_absent_module" in proc.stderr

    def test_missing_deps_install_into_project_venv(self):
        bat = self._bat()
        # 缺依赖 -> 建项目内 .venv，而不是塞进用户已有解释器
        assert "%PY% -m venv .venv" in bat
        assert ".venv\\Scripts\\python.exe" in bat
        # 建不出来时要明确告警并回落到原解释器，不能静默继续
        assert "[WARN] could not create .venv" in bat
        # 逃生门：想装进现有解释器的人要有办法
        assert "LPR_NO_VENV" in bat

    def test_deps_are_reprobed_after_install(self):
        """装完必须复检 —— "装成功但装进别的解释器"不能静默通过。"""
        bat = self._bat()
        assert bat.count('-c "%PROBE%"') >= 2
        assert "[ERROR] Dependencies still missing after install." in bat

    def test_gui_extra_matches_launcher_fallback(self):
        """pyproject 的 extras 与 start.bat 的兜底包名不能各说各话。"""
        toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        bat = self._bat()
        assert 'gui = ["PySide6-Essentials>=6.5"]' in toml
        assert 'deep = ["rapidocr-onnxruntime>=1.2"]' in toml
        # 兜底清单里的每个包都必须能在 pyproject 里找到出处
        assert '"PySide6-Essentials>=6.5"' in bat
        assert '"rapidocr-onnxruntime>=1.2"' in bat
        assert '"numpy>=1.24"' in bat and '"Pillow>=10.0"' in bat

    def test_launcher_prefers_pyproject_as_single_source_of_truth(self):
        """首选 editable 安装，pyproject 才是依赖的唯一出处。"""
        assert 'pip install -e ".[deep,gui]"' in self._bat()

    def test_installer_does_not_inherit_foreign_pythonpath(self):
        """pip 不能继承外部 PYTHONPATH。

        原因（实测踩到）：``PYTHONPATH`` 里任何一个目录都会让该目录下的
        ``sitecustomize.py`` 被**安装器自己的进程**加载。一旦它是某种
        沙箱 / 企业策略 shim，pip 解包时的 ``os.unlink`` 就会被劫持，
        抛出与依赖本身毫无关系的错误（我们这边真实复现过
        ``SystemExit: 1`` 从第三方 sitecustomize 里冒出来）。
        """
        bat = self._bat()
        assert 'set "KEEP_PYTHONPATH=%PYTHONPATH%"' in bat
        assert 'set "PYTHONPATH="' in bat
        # 用户原有的 PYTHONPATH 要在启动时还原，不能被悄悄丢掉
        assert "%KEEP_PYTHONPATH%" in bat
        # 且必须走 if/else —— 直接拼 "src;%KEEP_PYTHONPATH%" 在空值时会留下
        # 尾随分号，而 sys.path 里的空条目等价于把当前工作目录加进去
        assert "if defined KEEP_PYTHONPATH" in bat
        assert 'set "PYTHONPATH=%~dp0src;%PYTHONPATH%"' not in bat


@pytest.mark.parametrize("doc", ["README.md", "docs/UPGRADE.md"])
def test_docs_mention_python_requirement(doc: str):
    """系统/环境要求必须写在文档里 —— 这是用户最常问的第一个问题。"""
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert "3.10" in text
