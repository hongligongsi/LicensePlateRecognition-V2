"""启动入口与运行环境兼容性的回归测试。

覆盖五件容易"改着改着就退化"的事：

1. ``python -m lpr`` 入口必须存在且能把参数透传给 CLI；
2. 摄像头后端不能又变回无条件的 Windows 专有 ``CAP_DSHOW``；
3. ``start.bat`` 必须是**纯 ASCII** —— 见 ``TestWindowsLauncher`` 的说明，
   这不是风格问题，批处理文件里出现多字节字符会让 cmd 读错行；
4. ``tools/launcher.py`` 只能依赖标准库，且"缺依赖就建项目内 .venv"这条
   策略不能被改回去（v2.3.5 修过一次）；
5. 中文提示必须真的打得出来（v2.3.6 的教训：写得对 ≠ cmd 会照常执行）。
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
LAUNCHER_SRC = (TOOLS / "launcher.py").read_text(encoding="utf-8")


def _load_launcher():
    """按文件路径加载 ``tools/launcher.py``（它不是包的一部分）。"""
    spec = importlib.util.spec_from_file_location("_lpr_launcher", TOOLS / "launcher.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load_launcher()


def _project_version() -> str:
    """取 ``pyproject.toml`` 里 ``[project]`` 段的 ``version``。

    刻意**不用** ``tomllib``：它是 3.11 才进标准库的，而 CI 的测试矩阵覆盖
    3.10（v2.4.3 的教训：3.10 上直接 ``ModuleNotFoundError``），为一句断言
    引入 ``tomli`` 回退依赖不划算。这里只按行切出 ``[project]`` 段取值，
    粒度足够且不依赖 TOML 解析器。
    """
    lines = (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("[project]")
    except ValueError:  # pragma: no cover - 正常仓库必含该段
        raise AssertionError("pyproject.toml 缺少 [project] 段") from None
    for line in lines[start + 1 :]:
        if line.startswith("["):  # 下一个段头，[project] 段结束
            break
        key, sep, value = line.partition("=")
        if sep and key.strip() == "version":
            return value.split("#", 1)[0].strip().strip('"')
    raise AssertionError("pyproject.toml 的 [project] 段缺少 version")


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
    """``start.bat`` 是一个纯 ASCII 引导壳。

    v2.3.6 的实测教训：**批处理文件里不能出现汉字。**

    cmd.exe 计算「下一行」的偏移时，拿上一行的**字符数**当**字节偏移**用。
    文件里只要有一个多字节字符，下一行就会被超前 ``(字节数 - 字符数)`` 个
    字节开始读。实测一行 20 个汉字的提示，让下一行开头 40 字节被吞掉，
    剩下的残片被当成命令执行：

    .. code-block:: text

               完成 —— CV 引擎、深度引擎、图形界面均已就绪。
        '，未改动你系统里的' 不是内部或外部命令，也不是可运行的程序。

    最坑的是它**不稳定**：短文件里超前量常常刚好落在下一行的缩进空格里，
    看上去一切正常 —— 我们自己的 6 行探针就这么骗过去一次。
    """

    @staticmethod
    def _raw() -> bytes:
        return (ROOT / "start.bat").read_bytes()

    @staticmethod
    def _text() -> str:
        return (ROOT / "start.bat").read_bytes().decode("ascii")

    def test_exists_and_is_ascii_only(self):
        # 一旦这里抛 UnicodeDecodeError，就是上面那段灾难的入口
        self._raw().decode("ascii")

    def test_has_no_bom(self):
        """BOM 会被 cmd 当成第一条命令的一部分。"""
        assert not self._raw().startswith(b"\xef\xbb\xbf")

    def test_keeps_crlf(self):
        """CRLF 是批处理必需的；这里确认没被误转成 LF。"""
        raw = self._raw()
        assert b"\r\n" in raw
        assert raw.count(b"\n") == raw.count(b"\r\n")

    def test_delegates_all_logic_to_python(self):
        text = self._text()
        assert "tools\\launcher.py" in text
        # 这些逻辑一个都不许长回批处理里（v2.3.6 全部搬进 launcher.py）
        for moved in ('set "PROBE=', "-m venv", "-m pip", "sys.version_info"):
            assert moved not in text, f"{moved!r} 应该留在 tools/launcher.py"

    def test_falls_back_to_bundled_notice_without_python(self):
        text = self._text()
        # 找不到任何解释器时用 `type` 打印 UTF-8 文本文件：批处理解析器全程
        # 不接触多字节字符，命令本身仍然是 ASCII
        assert "tools\\no-python.txt" in text
        # 控制台代码页 65001 是那个文件能被正确渲染的前提
        assert "chcp 65001" in text


class TestPythonLauncher:
    """``tools/launcher.py`` —— 中文提示与安装逻辑都住在这里。"""

    def test_imports_stdlib_only(self):
        """它必须能在"什么第三方包都没装"的解释器上跑起来。

        启动器的职责就是把环境补齐；一旦它 import 了 numpy / cv2 / PySide6，
        就会在**最需要它工作的那台机器**上 ImportError。
        """
        tree = ast.parse(LAUNCHER_SRC)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        assert "__future__" in imported  # sanity：AST 确实解析到了导入
        unknown = imported - set(sys.stdlib_module_names) - {"__future__"}
        assert not unknown, f"启动器引入了非标准库依赖：{sorted(unknown)}"
        assert "lpr" not in imported, "启动器不能依赖本项目（它可能还没装上）"

    def test_probe_covers_core_runtime_not_only_pyside6(self):
        # 只查 PySide6 是 v2.3.5 的原始 bug：干净解释器里 numpy/cv2 也没有
        for mod in ("numpy", "cv2", "PIL", "PySide6"):
            assert mod in LAUNCHER.PROBE, f"依赖探测漏了 {mod}"
        compile(LAUNCHER.PROBE, "<probe>", "exec")  # 会被拼进 -c 参数

    def test_probe_passes_here_and_fails_on_a_missing_module(self, monkeypatch):
        for mod in ("numpy", "cv2", "PIL", "PySide6"):
            pytest.importorskip(mod)
        assert LAUNCHER.probe([sys.executable]) is True
        monkeypatch.setattr(LAUNCHER, "PROBE", LAUNCHER.PROBE + ", lpr_definitely_absent")
        assert LAUNCHER.probe([sys.executable]) is False

    def test_probe_is_false_for_a_broken_interpreter(self):
        """路径不存在也不能抛异常 —— 上层只想知道"能不能用"。"""
        assert LAUNCHER.probe([str(ROOT / "no" / "such" / "python.exe")]) is False

    def test_version_matches_pyproject_cli_and_package(self):
        """版本号只有 ``lpr.__version__`` 一个字面量来源，其余全部对齐。

        v2.3.9 之前 cli.py 自己写一份、``lpr.__version__`` 又停在 2.3.1，
        状态栏、命令行 --version、pyproject 三处各说各话——用值比较钉死。
        """
        import lpr

        pkg_version = lpr.__version__
        assert pkg_version == LAUNCHER.VERSION
        assert _project_version() == pkg_version

        from lpr import cli

        assert pkg_version == cli.VERSION  # 不再是独立字面量，而是 from . import __version__
        source = (SRC / "lpr" / "cli.py").read_text(encoding="utf-8")
        assert "from . import __version__" in source
        assert f'__version__ = "{pkg_version}"' in (SRC / "lpr" / "__init__.py").read_text(
            encoding="utf-8"
        )

    def test_choose_target_reuses_a_complete_interpreter(self, tmp_path):
        target, action = LAUNCHER.choose_target(["py"], tmp_path, probe_fn=lambda _i: True)
        assert action == "reuse"
        assert target == ["py"]

    def test_choose_target_prefers_an_existing_venv(self, tmp_path):
        vpy = LAUNCHER.venv_python(tmp_path)
        vpy.parent.mkdir(parents=True)
        vpy.write_bytes(b"")
        target, action = LAUNCHER.choose_target(["py"], tmp_path, probe_fn=lambda _i: False)
        assert action == "use_venv"
        assert target == [str(vpy)]

    def test_choose_target_creates_a_project_local_venv(self, tmp_path):
        target, action = LAUNCHER.choose_target(["py"], tmp_path, probe_fn=lambda _i: False)
        assert action == "create_venv"
        # 关键：目标是项目内 .venv，绝不是用户已有的那个解释器
        assert target == [str(LAUNCHER.venv_python(tmp_path))]
        assert target != ["py"]

    def test_choose_target_honours_the_no_venv_escape_hatch(self, tmp_path):
        target, action = LAUNCHER.choose_target(
            ["py"], tmp_path, probe_fn=lambda _i: False, no_venv=True
        )
        assert action == "global"
        assert target == ["py"]

    def test_venv_lives_inside_the_project(self):
        assert LAUNCHER.venv_python(ROOT).parent.parent == ROOT / ".venv"

    def test_install_env_drops_a_foreign_pythonpath(self):
        """pip 不能继承外部 PYTHONPATH。

        外部 PYTHONPATH 上的任何目录都会被 Python 自动 import 其中的
        ``sitecustomize.py``，等于让第三方代码在**安装器自己的进程**里执行。
        我们真实撞到过：pip 解包时的 ``os.unlink`` 被某个策略型 sitecustomize
        劫持，抛出与被装依赖毫无关系的 ``SystemExit: 1``。
        """
        env = LAUNCHER.install_env({"PATH": "x", "PYTHONPATH": "C:/somewhere/shim"})
        assert "PYTHONPATH" not in env
        assert env["PATH"] == "x"

    def test_build_env_injects_src_without_a_trailing_separator(self, tmp_path):
        # 空值直接拼字符串会留下尾随分隔符，而 sys.path 里的空条目等价于
        # 把当前工作目录加进去
        env = LAUNCHER.build_env({}, tmp_path)
        assert env["PYTHONPATH"] == str(tmp_path / "src")
        assert not env["PYTHONPATH"].endswith(os.pathsep)

    def test_build_env_keeps_the_users_pythonpath(self, tmp_path):
        env = LAUNCHER.build_env({"PYTHONPATH": "D:/mine"}, tmp_path)
        assert env["PYTHONPATH"] == str(tmp_path / "src") + os.pathsep + "D:/mine"

    def test_install_spec_and_fallbacks_match_pyproject(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'gui = ["PySide6-Essentials>=6.5"]' in pyproject
        # 检查 deep 行包含 rapidocr 约束（带上界 1.3，避免 1.4.x 丢失快路径）
        assert "rapidocr-onnxruntime>=1.2" in pyproject
        assert LAUNCHER.INSTALL_SPEC == ".[deep,gui]"
        joined = " ".join(LAUNCHER.FALLBACK_PACKAGES)
        for pin in (
            "PySide6-Essentials>=6.5",
            "rapidocr-onnxruntime>=1.2,<1.3",
            "numpy>=1.24",
            "Pillow>=10.0",
        ):
            assert pin in joined, f"兜底清单漏了 {pin}"

    def test_main_rejects_python_below_the_floor(self, monkeypatch, capsys):
        monkeypatch.setattr(LAUNCHER, "MIN_PY", (99, 0))
        assert LAUNCHER.main(["engines"]) == 1
        out = capsys.readouterr().out
        assert "错误" in out  # 中文提示确实发出来了
        assert "99.0" in out


class TestLauncherMessages:
    """中文提示的落点：``tools/no-python.txt`` 与真实运行结果。"""

    @staticmethod
    def _raw() -> bytes:
        return (ROOT / "tools" / "no-python.txt").read_bytes()

    def test_notice_is_utf8_chinese_without_bom(self):
        raw = self._raw()
        assert not raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8")
        assert sum(1 for char in text if ord(char) > 127) > 50
        assert "python.org" in text
        assert "3.10" in text

    def test_notice_uses_crlf(self):
        """必须是 CRLF：``type`` 只吐字节，LF 行尾在 cmd 里不会回车，
        会显示成阶梯状错位。``.gitattributes`` 里也钉了这一条，防止换个
        平台检出就被转成 LF。"""
        raw = self._raw()
        assert raw.count(b"\n") == raw.count(b"\r\n")
        assert b"\r\n" in raw

    @pytest.mark.skipif(os.name != "nt", reason="cmd.exe 专有")
    def test_type_copies_the_notice_bytes_verbatim(self):
        """``type`` 把文件字节原样吐出来，UTF-8 中文不会在路上被改坏。

        这条是"无 Python 兜底路径"能成立的前提：没有解释器时，批处理只能靠
        `type` 输出那个提示文件；能不能*看得见*中文另说，取决于控制台代码页
        （start.bat 里有 `chcp 65001`），但字节层必须完整。

        命令行刻意用相对路径且不加引号 —— ``cmd /c`` 对嵌套引号有自己的解析
        规则，带引号的绝对路径会被拆坏（这条测试第一版就这么失败过）。
        """
        proc = subprocess.run(
            ["cmd", "/c", "type", r"tools\no-python.txt"],
            cwd=str(ROOT),
            capture_output=True,
            timeout=60,
        )
        assert "未找到 Python 解释器" in proc.stdout.decode("utf-8")

    @pytest.mark.skipif(os.name != "nt", reason="start.bat 是 Windows 批处理")
    def test_start_bat_really_launches_and_speaks_chinese(self):
        """端到端：真跑一次 ``start.bat``，确认它没被 cmd 读错行。

        这是唯一能抓住 v2.3.6 那个 bug 的测试 —— 静态检查会告诉你"文件写得
        对"，但只有真跑一遍才能知道 cmd 会不会把中文残片当命令执行。
        """
        for mod in ("numpy", "cv2", "PIL", "PySide6"):
            pytest.importorskip(mod)
        env = dict(os.environ)
        env["LPR_PYTHON"] = sys.executable  # 复用当前解释器，避免真去装东西
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            ["cmd", "/c", "start.bat engines"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            timeout=300,
        )
        out = proc.stdout.decode("utf-8", "replace")
        assert proc.returncode == 0, out + proc.stderr.decode("utf-8", "replace")
        assert "解释器正常" in out
        assert "正在启动" in out
        # 中文被当成命令执行时的典型症状
        assert "不是内部或外部命令" not in out


@pytest.mark.parametrize("doc", ["README.md", "docs/UPGRADE.md"])
def test_docs_mention_python_requirement(doc: str):
    """系统/环境要求必须写在文档里 —— 这是用户最常问的第一个问题。"""
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert "3.10" in text
