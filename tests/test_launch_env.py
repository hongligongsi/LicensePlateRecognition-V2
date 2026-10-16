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
    """按文件路径加载 ``tools/launcher.py``（它不是包的一部分）。

    加载前必须把模块放进 ``sys.modules``：``spec_from_file_location`` +
    ``exec_module`` 这条路径**不会**自动注册，而 ``@dataclass`` 这类装饰器会在
    装饰期去 ``sys.modules[cls.__module__].__dict__`` 取命名空间，漏了这一步
    就报 ``AttributeError: 'NoneType' object has no attribute '__dict__'``。
    （实测踩到过。启动器因此只用 ``NamedTuple``；这里也按标准写法补上。）
    """
    spec = importlib.util.spec_from_file_location("_lpr_launcher", TOOLS / "launcher.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_lpr_launcher"] = module
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
        # deep 行必须钉死 rapidocr 版本：1.2.13 的随包模型会把样张的 V 读成 9
        assert "rapidocr-onnxruntime==1.2.3" in pyproject
        assert LAUNCHER.INSTALL_SPEC == ".[deep,gui]"
        joined = " ".join(LAUNCHER.FALLBACK_PACKAGES)
        for pin in (
            "PySide6-Essentials>=6.5",
            "rapidocr-onnxruntime==1.2.3",
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


def _fake_root(tmp_path: Path, *, svm: bool = True, yolo: bool = True) -> Path:
    """造一个假的「项目根」，用来在**不碰真实 models/** 的前提下验证自检。"""
    root = tmp_path / "proj"
    (root / "models" / "yolo").mkdir(parents=True)
    if svm:
        (root / "models" / "svm.dat").touch()
        (root / "models" / "svmchinese.dat").touch()
    if yolo:
        (root / "models" / "yolo" / "plate_det.onnx").touch()
    return root


def _all_backends(**overrides: bool) -> dict[str, bool]:
    """默认「后端齐全」，可覆盖个别项。"""
    payload = {"rapidocr_onnxruntime": True, "paddleocr": False, "onnxruntime": True}
    payload.update(overrides)
    return payload


class TestAssetManifest:
    """启动器的模型清单不能和引擎实现各说各话。

    这组测试防的是本模块最讨厌的那种 bug：启动器只允许依赖标准库，因此不可能
    ``import lpr`` 去问引擎「你需要哪些文件」，只能自带一份清单。清单一旦漂移，
    启动器就变成第二个「假真相」—— 它说齐全、引擎说不可用，而用户不知道该信谁。
    所以这里拿真实引擎的常量逐条钉死。
    """

    @staticmethod
    def _rel(path: Path) -> str:
        return path.relative_to(ROOT).as_posix()

    @staticmethod
    def _by_rel() -> dict[str, object]:
        return {asset.rel: asset for asset in LAUNCHER.ASSETS}

    def test_cv_paths_follow_config_models_dir(self):
        from lpr.config import MODELS_DIR

        by_rel = self._by_rel()
        cv_src = (SRC / "lpr" / "engines" / "cv_engine.py").read_text(encoding="utf-8")
        for name in ("svm.dat", "svmchinese.dat"):
            assert self._rel(MODELS_DIR / name) in by_rel, f"清单漏了 cv 需要的 {name}"
            # 文件名是写死在 cv_engine 里的字面量（那边没有导出常量）
            assert f'"{name}"' in cv_src, f"cv_engine 已经不再引用 {name}，清单要跟着改"

    def test_yolo_paths_follow_yolo_engine_constants(self):
        from lpr.engines import yolo_engine

        by_rel = self._by_rel()
        assert yolo_engine.locate_model_dir() == ROOT / yolo_engine.DEFAULT_MODEL_DIR
        cases = [
            (yolo_engine.DETECT_MODEL_NAME, False),
            (yolo_engine.RECOGNIZE_MODEL_NAME, True),
            (yolo_engine.CHARSET_NAME, True),
        ]
        for name, optional in cases:
            rel = self._rel(yolo_engine.locate_model_dir() / name)
            assert rel in by_rel, f"清单漏了 {name}"
            assert by_rel[rel].optional is optional, f"{name} 的必需/可选标错了"

    def test_detect_weight_is_the_only_required_yolo_asset(self):
        required = {a.rel for a in LAUNCHER.ASSETS if a.engine == "yolo" and not a.optional}
        assert required == {"models/yolo/plate_det.onnx"}

    def test_every_required_asset_carries_a_fix_hint(self):
        # 报了「缺什么」却不说「怎么办」，等于把问题丢回给用户
        for asset in LAUNCHER.ASSETS:
            if asset.optional:
                continue
            assert asset.hint.strip(), f"{asset.rel} 缺少修复提示"

    def test_fusion_owns_no_weights(self):
        # 三合一的可用性由成员派生，它自己不直接吃权重
        assert all(a.engine in {"cv", "deep", "yolo"} for a in LAUNCHER.ASSETS)

    def test_engine_names_and_labels_match_the_registry(self):
        from lpr.engines import available_engines, describe_engines

        assert set(LAUNCHER.ENGINE_ORDER) == set(available_engines())
        assert set(LAUNCHER.ENGINE_LABELS) == set(LAUNCHER.ENGINE_ORDER)
        described = {row["name"]: row["display_name"] for row in describe_engines()}
        assert {name: LAUNCHER.ENGINE_LABELS[name] for name in described} == described

    def test_backend_probes_match_the_engines_own_probes(self):
        """后端模块名必须就是引擎 ``is_available()`` 里 import 的那个名字。"""
        deep_src = (SRC / "lpr" / "engines" / "deep_engine.py").read_text(encoding="utf-8")
        yolo_src = (SRC / "lpr" / "engines" / "yolo_engine.py").read_text(encoding="utf-8")
        for module in LAUNCHER.BACKENDS:
            assert f"import {module}" in deep_src + yolo_src, f"{module} 在引擎里已经不存在"
        assert LAUNCHER.BACKENDS["onnxruntime"][0] == "yolo"
        assert LAUNCHER.BACKENDS["rapidocr_onnxruntime"][0] == "deep"

    def test_engine_level_backend_needs_cover_the_same_modules(self):
        """``BACKEND_NEEDS`` 与 ``BACKENDS`` 是同一批模块的两种视角，不许漂移。"""
        needed = {module for modules, _ in LAUNCHER.BACKEND_NEEDS.values() for module in modules}
        assert needed == set(LAUNCHER.BACKENDS)
        for engine, (modules, _what) in LAUNCHER.BACKEND_NEEDS.items():
            for module in modules:
                assert LAUNCHER.BACKENDS[module][0] == engine, f"{module} 的归属对不上 {engine}"


class TestVerdictAgainstEngines:
    """静态判定不许与引擎自己的 ``is_available()`` 打架。"""

    @staticmethod
    def _static() -> dict[str, object]:
        runtime_ok, _gui, backends = LAUNCHER.module_report([sys.executable])
        states = LAUNCHER.scan_assets(ROOT)
        return {
            s.name: s
            for s in LAUNCHER.engine_report(states, runtime_ok=runtime_ok, backends=backends)
        }

    @staticmethod
    def _real() -> dict[str, bool]:
        from lpr.engines import describe_engines

        return {row["name"]: bool(row["available"]) for row in describe_engines()}

    def test_never_more_optimistic_than_the_engines_themselves(self):
        """启动器说「可用」的，引擎自己也必须说可用 —— 这是用户唯一会看到的那句话。"""
        real = self._real()
        for name, status in self._static().items():
            if status.ok:
                assert real[name], f"启动器报 {name} 可用，引擎自己报不可用"

    def test_exact_agreement_when_everything_is_provisioned(self):
        real = self._real()
        if not all(real.values()):
            pytest.skip("本机有引擎不可用（CI 只装 .[dev]），一致性留待齐全的机器验证")
        assert {name: s.ok for name, s in self._static().items()} == real


class TestEngineReport:
    """由「运行库 + 后端 + 权重」推可用性的规则本身。"""

    def test_all_present_means_all_available(self, tmp_path):
        states = LAUNCHER.scan_assets(_fake_root(tmp_path))
        report = {
            s.name: s
            for s in LAUNCHER.engine_report(states, runtime_ok=True, backends=_all_backends())
        }
        assert all(s.ok for s in report.values())
        assert report["fusion"].detail == "成员：yolo / deep / cv"

    def test_missing_detection_weight_only_breaks_yolo(self, tmp_path):
        states = LAUNCHER.scan_assets(_fake_root(tmp_path, yolo=False))
        report = {
            s.name: s
            for s in LAUNCHER.engine_report(states, runtime_ok=True, backends=_all_backends())
        }
        assert not report["yolo"].ok
        assert "plate_det.onnx" in report["yolo"].reason
        assert "fetch_yolo_weights.py" in report["yolo"].fix
        assert report["cv"].ok and report["deep"].ok
        # 三合一退到只剩两个成员，但仍然可用
        assert report["fusion"].ok
        assert "deep" in report["fusion"].detail and "yolo" not in report["fusion"].detail

    def test_yolo_reports_backend_and_weight_together(self, tmp_path):
        """缺后端 + 缺权重是一次报全 —— 否则用户要装一次、重启、再发现还缺东西。"""
        states = LAUNCHER.scan_assets(_fake_root(tmp_path, yolo=False))
        report = {
            s.name: s
            for s in LAUNCHER.engine_report(
                states, runtime_ok=True, backends=_all_backends(onnxruntime=False)
            )
        }
        assert "onnxruntime" in report["yolo"].reason
        assert "plate_det.onnx" in report["yolo"].reason
        assert report["yolo"].fix.count("\n") >= 2

    def test_missing_svm_weight_breaks_cv_only(self, tmp_path):
        states = LAUNCHER.scan_assets(_fake_root(tmp_path, svm=False))
        report = {
            s.name: s
            for s in LAUNCHER.engine_report(states, runtime_ok=True, backends=_all_backends())
        }
        assert not report["cv"].ok
        assert "svm.dat" in report["cv"].reason
        assert report["deep"].ok and report["yolo"].ok

    def test_no_runtime_library_fails_every_engine_with_one_reason(self, tmp_path):
        states = LAUNCHER.scan_assets(_fake_root(tmp_path))
        report = list(LAUNCHER.engine_report(states, runtime_ok=False, backends=_all_backends()))
        assert not any(s.ok for s in report)
        for status in report:
            assert "运行库" in status.reason or "可用成员" in status.reason

    def test_deep_accepts_paddleocr_as_an_alternative_backend(self, tmp_path):
        states = LAUNCHER.scan_assets(_fake_root(tmp_path))
        report = {
            s.name: s
            for s in LAUNCHER.engine_report(
                states,
                runtime_ok=True,
                backends=_all_backends(rapidocr_onnxruntime=False, paddleocr=True),
            )
        }
        assert report["deep"].ok

    def test_optional_assets_never_break_anything(self, tmp_path):
        # 只有必需资产存在（可选的两个都没有），四个引擎依然全绿
        root = _fake_root(tmp_path)
        assert not (root / "models" / "yolo" / "plate_rec.onnx").exists()
        states = LAUNCHER.scan_assets(root)
        assert [s.asset.rel for s in states if s.missing] == []
        assert all(
            s.ok for s in LAUNCHER.engine_report(states, runtime_ok=True, backends=_all_backends())
        )

    def test_undersized_asset_is_flagged_but_not_failed(self, tmp_path):
        """桩模型是合法用法，所以「文件偏小」只能提示，不能判失败。"""
        root = _fake_root(tmp_path)
        (root / "models" / "yolo" / "plate_det.onnx").write_bytes(b"x" * 604)
        states = LAUNCHER.scan_assets(root)
        states2 = LAUNCHER.scan_assets(root)
        assert [s.undersized for s in states].count(True) == 1
        assert "可能是自检桩模型" in "\n".join(LAUNCHER.render_assets(states2))
        assert all(
            s.ok for s in LAUNCHER.engine_report(states, runtime_ok=True, backends=_all_backends())
        )


class TestProbeModules:
    """一次子进程问清多个模块 —— 真 import，不是 find_spec。"""

    def test_reports_true_and_false_per_module(self):
        result = LAUNCHER.probe_modules([sys.executable], ("json", "lpr_definitely_absent"))
        assert result == {"json": True, "lpr_definitely_absent": False}

    def test_broken_interpreter_yields_all_false(self):
        missing = str(ROOT / "no" / "such" / "python.exe")
        assert LAUNCHER.probe_modules([missing], ("json",)) == {"json": False}

    def test_empty_request_returns_empty(self):
        assert LAUNCHER.probe_modules([sys.executable], ()) == {}

    def test_uses_exactly_one_subprocess_for_all_modules(self, monkeypatch):
        calls = []
        real_run = subprocess.run

        def spy(cmd, *args, **kwargs):
            calls.append(cmd)
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(LAUNCHER.subprocess, "run", spy)
        LAUNCHER.probe_modules([sys.executable], ("json", "sys", "os"))
        assert len(calls) == 1, "每个模块各起一个进程会让每次启动多等好几秒"


class TestDisplayWidth:
    """中文按 2 列算 —— 不然引擎表的「状态」列会对不齐。"""

    def test_cjk_counts_as_two_columns(self):
        assert LAUNCHER._width("传统 CV 引擎") == 12
        assert LAUNCHER._width("abc") == 3

    def test_pad_produces_equal_rendered_width(self):
        rows = [LAUNCHER._pad("传统 CV 引擎", 14), LAUNCHER._pad("检测识别引擎", 14)]
        assert {LAUNCHER._width(r) for r in rows} == {14}


class TestDoctor:
    """``doctor`` 只体检：不建 venv、不装包、不改动任何文件。"""

    @staticmethod
    def _patch_probes(monkeypatch, runtime: bool = True, gui: bool = True, backends=None):
        payload = _all_backends() if backends is None else backends

        def fake(_interp, modules, timeout=180):
            return {
                **dict.fromkeys(LAUNCHER.RUNTIME_MODULES, runtime),
                **dict.fromkeys(LAUNCHER.GUI_MODULES, gui),
                **{m: bool(payload.get(m)) for m in LAUNCHER.BACKENDS},
            }

        monkeypatch.setattr(LAUNCHER, "probe_modules", fake)

    def _forbid_side_effects(self, monkeypatch):
        def boom(*_args, **_kwargs):
            raise AssertionError("doctor 是只读的，不该走到这里")

        for name in ("create_venv", "install_deps", "launch"):
            monkeypatch.setattr(LAUNCHER, name, boom)

    def test_changes_nothing_on_disk_and_touches_no_installer(self, tmp_path, monkeypatch, capsys):
        self._forbid_side_effects(monkeypatch)
        self._patch_probes(monkeypatch)
        root = _fake_root(tmp_path, svm=False, yolo=False)
        before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
        LAUNCHER.doctor(["py"], "create_venv", ["py"], root)
        after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
        assert before == after
        out = capsys.readouterr().out
        assert "环境体检" in out and "只读" in out

    def test_exit_code_zero_only_when_every_engine_is_available(
        self, tmp_path, monkeypatch, capsys
    ):
        self._patch_probes(monkeypatch)
        ok_root = _fake_root(tmp_path / "ok")
        assert LAUNCHER.doctor(["py"], "reuse", ["py"], ok_root) == 0
        assert "全部可用" in capsys.readouterr().out

        broken_root = _fake_root(tmp_path / "broken", yolo=False)
        assert LAUNCHER.doctor(["py"], "reuse", ["py"], broken_root) == 1
        out = capsys.readouterr().out
        assert "检测识别引擎" in out
        assert "fetch_yolo_weights.py" in out
        assert "3/4" in out

    def test_notes_when_the_target_interpreter_does_not_exist_yet(
        self, tmp_path, monkeypatch, capsys
    ):
        self._patch_probes(monkeypatch)
        root = _fake_root(tmp_path)
        LAUNCHER.doctor(["/tmp/does-not-exist/python"], "create_venv", ["py"], root)
        out = capsys.readouterr().out
        assert "目标解释器尚不存在" in out

    def test_optional_missing_backend_is_not_marked_as_a_failure(
        self, tmp_path, monkeypatch, capsys
    ):
        """装了 rapidocr、没装 paddleocr 的机器不该看到一行刺眼的「不可用」。"""
        self._patch_probes(monkeypatch, backends=_all_backends(paddleocr=False))
        LAUNCHER.doctor(["py"], "reuse", ["py"], _fake_root(tmp_path))
        rows = [ln for ln in capsys.readouterr().out.splitlines() if "paddleocr" in ln]
        assert rows, "体检必须列出 paddleocr"
        assert rows[0].lstrip().startswith("[--]"), rows[0]
        assert "未安装不影响" in rows[0]


class TestStartupPreflight:
    """每次启动都报权重结论；但缺权重**不许**挡住启动。"""

    @staticmethod
    def _patch(monkeypatch, root: Path, backends=None):
        seen: dict[str, object] = {}
        payload = _all_backends() if backends is None else backends
        monkeypatch.setattr(LAUNCHER, "ROOT", root)
        monkeypatch.setattr(LAUNCHER, "probe", lambda interp, timeout=120: True)
        monkeypatch.setattr(LAUNCHER, "module_report", lambda target: (True, True, payload))

        def fake_launch(target, args, root_=LAUNCHER.ROOT):
            seen["args"] = list(args)
            return 0

        monkeypatch.setattr(LAUNCHER, "launch", fake_launch)
        return seen

    def test_reports_the_missing_weight_and_still_launches(self, tmp_path, monkeypatch, capsys):
        seen = self._patch(monkeypatch, _fake_root(tmp_path, yolo=False))
        assert LAUNCHER.main(["engines"]) == 0
        out = capsys.readouterr().out
        assert "[4/5] 模型权重与引擎" in out
        assert "不可用：检测识别引擎" in out
        assert "fetch_yolo_weights.py" in out
        assert "start.bat doctor" in out
        # 关键：仍然启动了，且参数原样透传
        assert "[5/5] 正在启动..." in out
        assert seen["args"] == ["engines"]

    def test_says_all_available_when_nothing_is_missing(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, _fake_root(tmp_path))
        LAUNCHER.main([])
        out = capsys.readouterr().out
        assert "全部可用：传统 CV 引擎 / 深度学习引擎 / 检测识别引擎 / 三合一引擎" in out

    def test_doctor_token_is_intercepted_and_never_reaches_the_cli(self, monkeypatch, capsys):
        calls: dict[str, object] = {}

        def fake_doctor(target, action, probe, root):
            calls["doctor"] = True
            return 0

        def fake_launch(target, args, root_=LAUNCHER.ROOT):
            calls["launch"] = True
            return 0

        monkeypatch.setattr(LAUNCHER, "choose_target", lambda *a, **k: (["py"], "reuse"))
        monkeypatch.setattr(LAUNCHER, "doctor", fake_doctor)
        monkeypatch.setattr(LAUNCHER, "launch", fake_launch)
        for token in ("doctor", "--doctor", "--check"):
            calls.clear()
            assert LAUNCHER.main([token]) == 0
            assert calls.get("doctor") is True, token
            assert "launch" not in calls, f"{token} 被当成引擎参数透传了"

    def test_cli_check_subcommand_is_not_swallowed(self, monkeypatch, capsys):
        """``lpr check 京A12345`` 是 CLI 子命令，不许被 doctor 的别名吃掉。"""
        seen: dict[str, object] = {}
        monkeypatch.setattr(LAUNCHER, "choose_target", lambda *a, **k: (["py"], "reuse"))
        monkeypatch.setattr(LAUNCHER, "doctor", lambda *a: pytest.fail("不该走 doctor"))
        monkeypatch.setattr(LAUNCHER, "scan_assets", lambda root=LAUNCHER.ROOT: [])
        monkeypatch.setattr(LAUNCHER, "module_report", lambda target: (True, True, _all_backends()))

        def fake_launch(target, args, root_=LAUNCHER.ROOT):
            seen["args"] = list(args)
            return 0

        monkeypatch.setattr(LAUNCHER, "launch", fake_launch)
        LAUNCHER.main(["check", "京A12345"])
        assert seen["args"] == ["check", "京A12345"]
        assert "全部可用" in capsys.readouterr().out

    def test_extra_arguments_after_doctor_are_ignored_with_a_hint(self, monkeypatch, capsys):
        monkeypatch.setattr(LAUNCHER, "choose_target", lambda *a, **k: (["py"], "reuse"))
        monkeypatch.setattr(LAUNCHER, "doctor", lambda *a: 0)
        assert LAUNCHER.main(["doctor", "predict", "a.jpg"]) == 0
        assert "忽略多余参数" in capsys.readouterr().out


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
