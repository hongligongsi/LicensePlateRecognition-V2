#!/usr/bin/env python3
"""车牌识别系统的启动器 —— ``start.bat`` 的 ASCII 引导壳把控制权交给这里。

**为什么启动逻辑在 Python 而不在 .bat？**

实测（v2.3.6）：cmd.exe 计算批处理「下一行」的偏移时，拿着上一行的**字符数**
当**字节偏移**用。文件里只要有一个多字节字符，下一行就会被超前
``(字节数 - 字符数)`` 个字节开始读。一行 20 个汉字的中文提示，让下一行开头
40 字节被吞掉，剩下的残片被当成命令执行：

.. code-block:: text

           完成 —— CV 引擎、深度引擎、图形界面均已就绪。
    '，未改动你系统里的' 不是内部或外部命令，也不是可运行的程序。

在短文件里这个超前量往往刚好落在下一行的缩进空格里，看上去"没问题"（我们
的小探针就这样骗过了自己一次），一旦提示变长就必炸。所以 ``start.bat``
保持**纯 ASCII**，所有中文都从本模块打印 —— Python 在 Windows 上经
``WriteConsoleW`` 输出，与控制台代码页（65001 / 936）无关。

**本模块只允许依赖标准库。** 它自己的职责就是"把运行环境补齐"，如果它反过来
依赖被补齐的环境（numpy / cv2 / PySide6），就会在一个干净的解释器上直接
ImportError —— 那正是它最需要工作的场景。``tests/test_launch_env.py``
有一条守卫测试盯着这一点。

环境变量：

* ``LPR_PYTHON``   —— 不读；由 ``start.bat`` 用来挑选解释器；
* ``LPR_NO_VENV``  —— 非空则不创建项目内 ``.venv``，直接装进当前解释器。

子命令：

* ``doctor``（别名 ``--doctor`` / ``--check``）—— **只体检，不启动**。把解释器、
  运行依赖、模型权重、四个引擎的可用性一次列全，每条不可用都附确切修复命令。
  它**只读**：不建 venv、不装包、不改任何文件，因此可以在环境已经坏掉时放心跑。
  四个引擎全部可用时返回码为 0，否则为 1（便于脚本判断）。
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

#: 与 pyproject.toml / cli.py / config.py 保持一致
VERSION = "2.7.3"
#: 低于此版本直接拒绝（Qt 6 与 typing 语法都需要）
MIN_PY = (3, 10)
#: 核心运行库 + 界面库，四项缺一不可
PROBE = "import numpy, cv2, PIL, PySide6"
#: 首次安装的 extras —— pyproject 是依赖的唯一出处
INSTALL_SPEC = ".[deep,gui]"
#: 可编辑安装跑不起来时的兜底清单（与 pyproject 的 extras 一一对应）
FALLBACK_PACKAGES = (
    "numpy>=1.24",
    "opencv-python-headless>=4.10,<5",
    "Pillow>=10.0",
    "rapidocr-onnxruntime==1.2.3",
    "PySide6-Essentials>=6.5",
)

ROOT = Path(__file__).resolve().parents[1]


# ────────────────────────────────────────────────────────────────
# 模型权重与引擎自检
#
# 「依赖装齐」不等于「能跑」。权重不进仓库（体积 + 许可，见
# ``models/yolo/README.md``），而**不同引擎缺的东西不一样**：``cv`` 缺
# ``models/*.dat`` 是加载期直接抛，``yolo`` 缺 ``plate_det.onnx`` 只是
# 「这个引擎不可用」。而 ``.[deep,gui]`` 这个默认安装规格**根本不含
# ``onnxruntime``** —— 所以默认装完之后 ``yolo`` 必然是「不可用」。
#
# 启动器原先只探测 numpy/cv2/PIL/PySide6，于是权重或后端缺失时它照样打印
# 「均已就绪」，用户在界面下拉框里看到的是那行「（不可用）」——反馈链断在这里。
#
# **清单必须与引擎实现一致**：路径写错的话，启动器就成了第二个「假真相」。
# ``tests/test_launch_env.py::TestAssetManifest`` 会拿真实引擎的模型路径逐条比对。
# ────────────────────────────────────────────────────────────────

#: 引擎运行库（界面库不在其中 —— 界面缺了不影响引擎可用性）
RUNTIME_MODULES = ("numpy", "cv2", "PIL")
#: 图形界面库，单独一档
GUI_MODULES = ("PySide6",)

#: 引擎运行库之外的后端 ``{模块名: (归属引擎, 说明, 修复命令)}``
BACKENDS: dict[str, tuple[str, str, str]] = {
    "rapidocr_onnxruntime": (
        "deep",
        "deep 的默认识别后端（推荐，轻量）",
        'pip install -e ".[deep]"',
    ),
    "paddleocr": ("deep", "deep 的备选识别后端（更大更慢）", 'pip install -e ".[deep-paddle]"'),
    "onnxruntime": ("yolo", "yolo 的唯一硬依赖（默认安装规格不含）", 'pip install -e ".[yolo]"'),
}

#: 引擎级的后端需求 ``{引擎: (候选模块, 用途)}`` —— 元组内是**任选其一**。
#: 必须与 :data:`BACKENDS` 覆盖同一批模块，``TestAssetManifest`` 有断言盯着。
BACKEND_NEEDS: dict[str, tuple[tuple[str, ...], str]] = {
    "deep": (("rapidocr_onnxruntime", "paddleocr"), "识别后端"),
    "yolo": (("onnxruntime",), "检测推理后端"),
}


class Asset(NamedTuple):
    """一个模型文件、它服务哪个引擎、缺了怎么办。

    用 ``NamedTuple`` 而不是 ``@dataclass``，理由很具体：本模块是按**文件路径**
    加载的（``start.bat`` 与 ``tests/test_launch_env.py::_load_launcher`` 都这么干），
    那条加载路径不把模块注册进 ``sys.modules``，而 ``@dataclass`` 装饰期要读
    ``sys.modules[cls.__module__].__dict__``，于是直接
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``。
    （实测：同一次加载里 ``NamedTuple`` / ``collections.namedtuple`` 都正常。）
    一个连自己都起不来的启动器，比不完善的启动器糟糕得多。
    """

    #: 相对项目根的路径
    rel: str
    #: 需要它的引擎
    engine: str
    #: 打印用的中文名
    label: str
    #: 缺失时的修复命令（原样打印，所以写成能直接粘进终端的形式）
    hint: str = ""
    #: 可选资产：缺了不算故障，引擎有回退路径
    optional: bool = False
    #: 小于该值就提示「可能是桩模型或下载不完整」——注意这只是提示，不是判定失败
    min_bytes: int = 1 << 10


#: 模型资产清单（与引擎实现一一对应，改这里必须同时改引擎或反之）
ASSETS: tuple[Asset, ...] = (
    Asset(
        rel="models/svm.dat",
        engine="cv",
        label="SVM 分类器（字母数字）",
        hint="随仓库分发，确认检出完整：git checkout -- models/svm.dat",
        min_bytes=1 << 10,
    ),
    Asset(
        rel="models/svmchinese.dat",
        engine="cv",
        label="SVM 分类器（中文省份）",
        hint="随仓库分发，确认检出完整：git checkout -- models/svmchinese.dat",
        min_bytes=1 << 10,
    ),
    Asset(
        rel="models/yolo/plate_det.onnx",
        engine="yolo",
        label="车牌检测网络（bbox + 四角点）",
        hint=(
            "python tools/fetch_yolo_weights.py --list          # 已知来源与许可\n"
            'python tools/fetch_yolo_weights.py --github <owner/repo> --asset "*det*.onnx"'
        ),
        min_bytes=1 << 16,
    ),
    Asset(
        rel="models/yolo/plate_rec.onnx",
        engine="yolo",
        label="车牌字符识别（可选）",
        hint="不装也能用：yolo 会复用 deep 引擎的识别器",
        optional=True,
    ),
    Asset(
        rel="models/yolo/charset.txt",
        engine="yolo",
        label="识别字符表（可选）",
        hint="只在自带 plate_rec.onnx 时才需要",
        optional=True,
    ),
)

#: 展示顺序（也是 ``fusion`` 成员自动退席的判定顺序）
ENGINE_ORDER = ("cv", "deep", "yolo", "fusion")
#: 引擎名的中文标签（与 ``lpr engines`` 的说法保持一致）
ENGINE_LABELS = {
    "cv": "传统 CV 引擎",
    "deep": "深度学习引擎",
    "yolo": "检测识别引擎",
    "fusion": "三合一引擎",
}

#: 只体检不启动的入口（只在参数**第一位**时拦截，避免和 CLI 子命令 ``check`` 混淆）
DOCTOR_TOKENS = frozenset({"doctor", "--doctor", "--check"})


# ────────────────────────────────────────────────────────────────
# 基础动作
# ────────────────────────────────────────────────────────────────
def probe_stmt(interp: Sequence[str], stmt: str, timeout: int = 120) -> bool:
    """跑一段 ``-c`` 语句，只关心返回码（静默）。"""
    try:
        proc = subprocess.run(
            [*interp, "-c", stmt],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def probe(interp: Sequence[str], timeout: int = 120) -> bool:
    """目标解释器是否已具备全部运行依赖（静默，只看返回码）。"""
    return probe_stmt(interp, PROBE, timeout)


#: 一次性问清多个模块能否 import 的小脚本。
#:
#: 为什么不用 ``importlib.util.find_spec``：find_spec 只找得到**文件**，
#: 装入一半（DLL 缺失、依赖不全）的包它照样返回路径，于是启动器报「可用」而
#: 引擎那边 ``is_available()`` 报「不可用」—— 正是本模块想消灭的那种
#: 「两个真相」。这里真去 ``__import__``，与引擎的判定口径一致。
#: 结果用带标记的前缀打印，避免被测模块自己往 stdout 写日志时把 JSON 污染。
_STATUS_SCRIPT = """
import json, sys
result = {}
for name in sys.argv[1:]:
    try:
        __import__(name)
        result[name] = True
    except Exception:
        result[name] = False
print("__LPR_STATUS__" + json.dumps(result))
"""


def probe_modules(
    interp: Sequence[str], modules: Iterable[str], timeout: int = 180
) -> dict[str, bool]:
    """在**一个**子进程里试 import 多个模块，返回 ``{模块名: 是否可用}``。

    合并成一次调用是刻意的：Windows 上解释器冷启动本身就要 ~0.5 s，
    逐个模块各起一个进程会让每次启动多等好几秒。真去 import 而不是查
    ``find_spec``，理由见 :data:`_STATUS_SCRIPT`。

    探测失败（解释器起不来、超时、输出被污染）时全部按 ``False`` 返回 ——
    对调用方来说「问不出来」和「没有」的处理方式一样：给提示让用户自己确认。
    """
    names = list(modules)
    fallback = dict.fromkeys(names, False)
    if not names:
        return fallback
    try:
        proc = subprocess.run(
            [*interp, "-c", _STATUS_SCRIPT, *names],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return fallback
    marker = "__LPR_STATUS__"
    payload = ""
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            payload = line[len(marker) :]
    if not payload:
        return fallback
    try:
        parsed = json.loads(payload)
    except ValueError:
        return fallback
    if not isinstance(parsed, dict):
        return fallback
    return {name: bool(parsed.get(name, False)) for name in names}


def venv_python(root: Path = ROOT) -> Path:
    """项目内 ``.venv`` 的解释器路径（跨平台）。"""
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def choose_target(
    interp: Sequence[str],
    root: Path = ROOT,
    probe_fn: Callable[[Sequence[str]], bool] = probe,
    no_venv: bool = False,
) -> tuple[list[str], str]:
    """决定依赖该装进哪个解释器。

    返回 ``(解释器 argv, 动作)``，动作为：

    ``reuse``
        当前解释器本来就齐全 —— 直接用，不为建 venv 而建 venv。
    ``use_venv``
        项目内已有 ``.venv`` —— 复用它。
    ``create_venv``
        缺依赖且没有 ``.venv`` —— 建一个项目内的，**不碰**用户的全局解释器。
    ``global``
        ``LPR_NO_VENV`` 显式要求装进当前解释器。
    """
    if probe_fn(interp):
        return list(interp), "reuse"
    local = venv_python(root)
    if local.is_file():
        return [str(local)], "use_venv"
    if not no_venv:
        return [str(local)], "create_venv"
    return list(interp), "global"


def create_venv(interp: Sequence[str], root: Path = ROOT) -> bool:
    """用 ``interp`` 在项目内创建 ``.venv``。"""
    try:
        proc = subprocess.run([*interp, "-m", "venv", str(root / ".venv")])
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def install_env(base: Mapping[str, str]) -> dict[str, str]:
    """安装 pip 依赖时用的环境：**剥掉 PYTHONPATH**。

    外部 PYTHONPATH 上的任何目录都会被 Python 自动 import 其中的
    ``sitecustomize.py``，等于让第三方代码在 pip 自己的进程里执行 —— 我们
    真实撞到过：pip 解包时的 ``os.unlink`` 被某个策略型 sitecustomize 劫持，
    抛出与被装依赖毫无关系的 ``SystemExit: 1``。
    """
    env = dict(base)
    env.pop("PYTHONPATH", None)
    return env


def build_env(base: Mapping[str, str], root: Path = ROOT) -> dict[str, str]:
    """真正启动程序时用的环境：注入 ``src``，并接回用户原有的 PYTHONPATH。

    用判断而不是直接拼接 —— 空值直接拼会留下尾随分隔符，而 ``sys.path`` 里的
    空条目等价于把当前工作目录加进去。
    """
    env = dict(base)
    src = str(root / "src")
    keep = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src + os.pathsep + keep if keep else src
    return env


def install_deps(target: Sequence[str], root: Path = ROOT) -> bool:
    """把依赖装进 ``target``（见 :func:`install_env` 为什么要清 PYTHONPATH）。"""
    env = install_env(os.environ)
    try:
        main = subprocess.run(
            [*target, "-m", "pip", "install", "-e", INSTALL_SPEC], cwd=str(root), env=env
        )
        if main.returncode == 0:
            return True
        print("        可编辑安装失败，改用显式包名重试...")
        retry = subprocess.run(
            [*target, "-m", "pip", "install", *FALLBACK_PACKAGES], cwd=str(root), env=env
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return retry.returncode == 0


def launch(target: Sequence[str], args: Sequence[str], root: Path = ROOT) -> int:
    """以 ``target`` 解释器启动具体功能（无参数默认开界面）。"""
    try:
        proc = subprocess.run(
            [*target, "-m", "lpr", *(args or ["gui"])],
            cwd=str(root),
            env=build_env(os.environ, root),
        )
    except (OSError, subprocess.SubprocessError):
        return 1
    return proc.returncode


# ────────────────────────────────────────────────────────────────
# 自检：权重 → 引擎
# ────────────────────────────────────────────────────────────────
class AssetState(NamedTuple):
    """一个资产在磁盘上的实际状态。"""

    asset: Asset
    path: Path
    exists: bool
    size: int

    @property
    def missing(self) -> bool:
        """必需资产缺失（可选资产缺了不算）。"""
        return not self.exists and not self.asset.optional

    @property
    def undersized(self) -> bool:
        """存在但明显偏小 —— 可能是桩模型或没下完。只是提示，不判失败。"""
        return self.exists and 0 < self.size < self.asset.min_bytes


class EngineStatus(NamedTuple):
    """一个引擎的可用性结论。"""

    name: str
    ok: bool
    #: 不可用原因（可用时为空串）
    reason: str = ""
    #: 修复命令（没有可执行的修复动作时为空串）
    fix: str = ""
    #: 可用时的补充说明（例如三合一引擎的实际成员）
    detail: str = ""

    @property
    def label(self) -> str:
        return ENGINE_LABELS.get(self.name, self.name)


def scan_assets(root: Path = ROOT) -> list[AssetState]:
    """看盘：清单里每个资产是否存在、多大。纯文件系统操作，不起进程。"""
    states: list[AssetState] = []
    for asset in ASSETS:
        path = root / asset.rel
        try:
            size = path.stat().st_size if path.is_file() else 0
            exists = path.is_file()
        except OSError:  # 权限/坏链接等：按「不存在」处理并在行内说明
            size, exists = 0, False
        states.append(AssetState(asset, path, exists, size))
    return states


def engine_report(
    states: Sequence[AssetState],
    *,
    runtime_ok: bool,
    backends: Mapping[str, bool],
) -> list[EngineStatus]:
    """由「运行库 + 后端 + 权重」推出四个引擎的可用性。

    判定口径刻意与引擎自己的 ``is_available()`` 对齐：

    * ``cv``  —— 运行库 + 两个 SVM 模型文件（``is_available()`` 只看 cv2，
      但 ``,load()` 缺 ``svm.dat`` 会直接抛，所以这里按「能真的加载」算）；
    * ``deep`` —— 运行库 + ``rapidocr_onnxruntime`` 或 ``paddleocr`` 二者其一；
    * ``yolo`` —— 运行库 + ``onnxruntime`` + ``plate_det.onnx``；
    * ``fusion`` —— ``yolo`` / ``deep`` / ``cv`` 里至少有一个可用。

    缺运行库时所有引擎一起判不可用（此时原因统一指向依赖，而不是逐个编故事）。
    """
    missing: dict[str, list[AssetState]] = {}
    for state in states:
        if state.missing:
            missing.setdefault(state.asset.engine, []).append(state)

    def weight_part(engine: str) -> tuple[str, str]:
        rows = missing.get(engine, [])
        if not rows:
            return "", ""
        names = "、".join(s.asset.rel for s in rows)
        return f"缺少模型权重：{names}", "\n".join(s.asset.hint for s in rows if s.asset.hint)

    def merge(parts: list[tuple[str, str]]) -> tuple[str, str]:
        """把若干 (原因, 修复) 片段并成一句。

        刻意**累加**而不是「报了第一个就收工」：``yolo`` 同时缺 ``onnxruntime``
        和 ``plate_det.onnx`` 是最常见的新手形态，一次把两件事都说了，
        用户才不用装一个、重启、再发现还缺一个。
        """
        reasons = [p[0] for p in parts if p[0]]
        fixes = [p[1] for p in parts if p[1]]
        return "；".join(reasons), "\n".join(fixes)

    def backend_part(engine: str) -> tuple[str, str]:
        """后端缺失的原因。原因里**必须写出模块名** —— 只说「缺少后端」，用户没法照着修。"""
        candidates, what = BACKEND_NEEDS[engine]
        if any(backends.get(name) for name in candidates):
            return "", ""
        # 刻意不叫 missing：外层那个 missing 是「缺的权重」，同名会读错变量
        absent = [name for name in candidates if not backends.get(name)]
        reason = f"未安装 {'／'.join(absent)}（{what}）"
        # 多个候选是「任选其一」，只给首选那条的安装命令，少点噪音
        return reason, BACKENDS[absent[0]][2]

    verdict: dict[str, EngineStatus] = {}

    if not runtime_ok:
        deps_fix = 'pip install -e ".[deep,gui]"'
        for name in ("cv", "deep", "yolo"):
            verdict[name] = EngineStatus(
                name, False, "核心运行库不全（numpy / OpenCV / Pillow）", deps_fix
            )
    else:
        cv_reason, cv_fix = merge([weight_part("cv")])
        verdict["cv"] = EngineStatus("cv", not cv_reason, cv_reason, cv_fix)

        deep_reason, deep_fix = merge([backend_part("deep")])
        verdict["deep"] = EngineStatus("deep", not deep_reason, deep_reason, deep_fix)

        yolo_reason, yolo_fix = merge([backend_part("yolo"), weight_part("yolo")])
        verdict["yolo"] = EngineStatus("yolo", not yolo_reason, yolo_reason, yolo_fix)

    members = [m for m in ("yolo", "deep", "cv") if verdict[m].ok]
    if members:
        verdict["fusion"] = EngineStatus("fusion", True, detail=f"成员：{' / '.join(members)}")
    else:
        verdict["fusion"] = EngineStatus(
            "fusion",
            False,
            "没有任何可用成员（需要 cv / deep / yolo 至少一个能跑）",
            verdict["deep"].fix or verdict["cv"].fix,
        )

    return [verdict[name] for name in ENGINE_ORDER]


def module_report(target: Sequence[str]) -> tuple[bool, bool, dict[str, bool]]:
    """探测 ``target`` 上的运行库、界面库与后端，供 :func:`engine_report` 使用。

    返回 ``(运行库齐全, 界面库齐全, {后端: 是否可用})``。
    """
    status = probe_modules(target, (*RUNTIME_MODULES, *GUI_MODULES, *BACKENDS))
    runtime_ok = all(status.get(m, False) for m in RUNTIME_MODULES)
    gui_ok = all(status.get(m, False) for m in GUI_MODULES)
    backends = {name: bool(status.get(name, False)) for name in BACKENDS}
    return runtime_ok, gui_ok, backends


def describe_local_plan(action: str) -> str:
    """把 :func:`choose_target` 的结论翻成一句人话。"""
    return {
        "reuse": "当前解释器已齐全，直接复用",
        "use_venv": "复用项目内已有的 .venv",
        "create_venv": "项目内尚无 .venv，将创建并安装依赖",
        "global": "LPR_NO_VENV 已设置，将装入当前解释器",
    }.get(action, action)


# ── 显示宽度：中文与全角符号占 2 列，否则表格会歪 ─────────────────
def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def render_assets(states: Sequence[AssetState]) -> list[str]:
    """资产逐行明细（供 doctor 用）。"""
    lines: list[str] = []
    for state in states:
        tag = "[ok]"
        if not state.exists:
            tag = "[--]" if state.asset.optional else "[!!]"
        elif state.undersized:
            tag = "[??]"
        size = f"{state.size / 1048576:.2f} MB" if state.exists else "未提供"
        head = f"  {tag} {_pad(state.asset.label, 30)} {_pad(size, 9)} {state.asset.rel}"
        lines.append(head.rstrip())
        if state.missing:
            lines.extend(f"        {row}" for row in state.asset.hint.splitlines())
        elif state.undersized:
            lines.append(
                f"        文件偏小（{state.size} 字节）—— 可能是自检桩模型"
                "（tools/make_stub_onnx.py）或下载不完整"
            )
    return lines or ["  （清单为空）"]


def render_engines(report: Sequence[EngineStatus]) -> list[str]:
    """引擎可用性逐行明细（同时用于 doctor 与每次启动）。"""
    lines: list[str] = []
    for status in report:
        head = f"  {_pad(status.label, 14)} {'可用' if status.ok else '不可用'}"
        if status.ok:
            lines.append(f"{head}（{status.detail}）" if status.detail else head)
            continue
        lines.append(f"{head} —— {status.reason}")
        fixes = [row for row in status.fix.splitlines() if row]
        for index, row in enumerate(fixes):
            # 只给第一条挂「修复：」，其余对齐悬挂缩进，多条修复才不像复读
            lines.append(f"        {'修复：' if index == 0 else '      '}{row}")
    return lines


def dep_rows(
    runtime_ok: bool,
    gui_ok: bool,
    backends: Mapping[str, bool],
    report: Sequence[EngineStatus],
) -> list[tuple[str, str, str]]:
    """``[(标记, 模块名, 说明)]`` —— 缺失的后端只有在**它拖垮了引擎**时才标 ``[!!]``。

    不这么分的话，装了 rapidocr 却没装 paddleocr 的机器会看到一行刺眼的
    「不可用」，而 deep 引擎其实好好的 —— 假警报比不报还糟。
    """
    verdict = {s.name: s for s in report}
    rows: list[tuple[str, str, str]] = []
    for name in RUNTIME_MODULES:
        rows.append(("[ok]" if runtime_ok else "[!!]", name, "运行库（必需）"))
    for name in GUI_MODULES:
        rows.append(("[ok]" if gui_ok else "[!!]", name, "图形界面（必需）"))
    for name, (owner, note, _fix) in BACKENDS.items():
        ok = bool(backends.get(name))
        if ok:
            rows.append(("[ok]", name, note))
            continue
        needed = not verdict[owner].ok
        mark = "[!!]" if needed else "[--]"
        rows.append((mark, name, f"{note} —— 未安装" if needed else f"{note}，未安装不影响"))
    return rows


def doctor(target: Sequence[str], action: str, probe_target: Sequence[str], root: Path) -> int:
    """只体检不启动：串起解释器 / 依赖 / 权重 / 引擎四段，返回码即结论。

    ``target`` 是将来真正跑程序的那个解释器，``probe_target`` 是实际拿来探测的
    （项目内 ``.venv`` 还不存在时退回到当前解释器）。**本函数不写任何文件**。
    """
    runtime_ok, gui_ok, backends = module_report(probe_target)
    states = scan_assets(root)
    report = engine_report(states, runtime_ok=runtime_ok, backends=backends)

    print("=" * 68)
    print(" 环境体检（只读：不建 .venv、不装包、不改动任何文件）")
    print("=" * 68)

    print()
    print("解释器")
    print(f"  当前      {sys.executable}  (Python {sys.version.split()[0]})")
    print(f"  将使用    {target[0]}")
    print(f"  动作      {describe_local_plan(action)}")
    if probe_target[0] != target[0]:
        print(f"  注：目标解释器尚不存在，以下依赖探测基于 {probe_target[0]}")

    print()
    print("运行依赖")
    for mark, name, note in dep_rows(runtime_ok, gui_ok, backends, report):
        ok = mark == "[ok]"
        state = "可用" if ok else ("未安装" if mark == "[--]" else "不可用")
        print(f"  {mark} {_pad(name, 22)} {_pad(state, 8)} {note}")
    if not runtime_ok:
        print('       修复：pip install -e ".[deep,gui]"（或在项目根跑 start.bat 自动装）')

    print()
    print("模型权重")
    for line in render_assets(states):
        print(line)

    print()
    print("引擎可用性")
    for line in render_engines(report):
        print(line)

    ok_count = sum(1 for s in report if s.ok)
    print()
    if ok_count == len(report):
        print(f"结论：{len(report)} 个引擎全部可用。")
        return 0
    print(f"结论：{ok_count}/{len(report)} 个引擎可用 —— 按上面各条的「修复」处理即可。")
    print("      缺权重/后端只影响对应引擎，其余引擎和图形界面照常可用。")
    return 1


# ────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────
def _make_pipes_utf8() -> None:
    """把**管道 / 重定向**的 stdout、stderr 拨到 UTF-8。

    控制台（``isatty()``）保持原编码 —— 中文 Windows 控制台是 cp936，本来就
    显示得出来。但管道/重定向下的编码由 locale 决定：CI 的 Windows runner 是
    cp1252，这里 ``print`` 一行中文就抛 ``UnicodeEncodeError``，启动器直接以
    非零码退出（v2.4.3 的 CI 教训）。``start.bat`` 自己不能碰编码 ——
    它必须保持纯 ASCII。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                continue
        except (AttributeError, ValueError):
            continue
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding == "utf8":
            continue
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


# ────────────────────────────────────────────────────────────────
# 运行日志的去处
# ────────────────────────────────────────────────────────────────

#: 读不出 ``config.json`` 时的兜底目录 —— 必须与 ``lpr.config.Config.log_dir``
#: 的默认值一致（有测试交叉校验），否则启动器就成了第二个「假真相」。
DEFAULT_LOG_DIR = "logs"


def _truthy(value: object) -> bool:
    """宽松地读 JSON/字符串布尔：``false`` / ``"0"`` / ``"off"`` 都算假。

    启动器不许 import ``lpr``（只依赖标准库），所以这几个字段的解析
    在 ``lpr.config`` 里另有一份 —— 用测试钉住两边一致。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def log_target(root: Path = ROOT) -> Path | None:
    """本项目的日志目录；配置里关掉了文件日志时返回 ``None``。

    刻意不猜：用户改过 ``config.json`` 就用他的值，只有读不出文件时才回落到
    与 ``Config`` 相同的默认目录。返回 ``None`` 时不打印任何"日志在哪儿"的
    提示 —— 说一个不存在的地方，比不说更糟。
    """
    try:
        raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return root / DEFAULT_LOG_DIR
    if not isinstance(raw, dict):
        return root / DEFAULT_LOG_DIR
    if not _truthy(raw.get("log_to_file", True)):
        return None
    name = str(raw.get("log_dir", "")).strip()
    return root / name if name else None


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    #: ``doctor`` 只在参数**第一位**时拦截：``lpr`` 自己有 ``check`` 子命令
    #: （校验车牌号），只认带横线的 ``--check`` 与独立的 ``doctor`` 才不会撞车。
    want_doctor = bool(args) and args[0] in DOCTOR_TOKENS
    if want_doctor:
        args = args[1:]

    _make_pipes_utf8()

    #: 重定向（``start.bat > out.txt``）时 stdout 会变成块缓冲，本模块的提示
    #: 会整段压在子进程输出之后。改成行缓冲，顺序在控制台和管道下才一致。
    #: 3.7 以前、或 stdout 被替换成不支持 reconfigure 的对象（pytest 捕获）
    #: 时直接跳过，不影响功能。
    with contextlib.suppress(AttributeError, OSError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    print("=" * 44)
    print(f" 车牌识别系统 v{VERSION}")
    print("=" * 44)
    print()

    if sys.version_info < MIN_PY:
        print(f"[错误] 需要 Python {MIN_PY[0]}.{MIN_PY[1]} 或更高版本。")
        print(f"        当前版本：{sys.version.split()[0]}  ({sys.executable})")
        return 1

    interp = [sys.executable]
    target, action = choose_target(interp, ROOT, no_venv=bool(os.environ.get("LPR_NO_VENV")))

    if want_doctor:
        if args:
            print(f"[提示] doctor 不启动程序，忽略多余参数：{' '.join(args)}")
            print()
        # 目标解释器还没建出来时，退回用当前解释器探测，并在报告里注明
        probe_target = target
        if action == "create_venv" and not Path(target[0]).is_file():
            probe_target = interp
        return doctor(target, action, probe_target, ROOT)

    print("[1/5] 解释器正常")
    print("       ", sys.executable)

    if action == "reuse":
        print("[2/5] 依赖齐全")
        print("[3/5] 无需安装")
    else:
        print("[2/5] 依赖缺失，正在准备运行环境...")
        print()
        if action == "use_venv":
            print("       使用已存在的 .venv")
        elif action == "create_venv":
            print("       正在创建 .venv ...")
            if create_venv(interp, ROOT):
                print("       已创建 .venv")
            else:
                target = interp
                print()
                print("[警告] 无法创建 .venv（venv 模块不可用？）。")
                print(f"       将回落到：{interp[0]}")
                print("       依赖会被安装进那个解释器。")
                print()
        else:  # global
            print(f"       LPR_NO_VENV 已设置，将安装到：{target[0]}")

        print(f"[3/5] 正在安装到 {target[0]}（仅首次，需要联网）...")
        print()
        if not install_deps(target, ROOT):
            print()
            print("[错误] 安装失败。请手动执行：")
            print(f'        {target[0]} -m pip install -e "{INSTALL_SPEC}"')
            return 1
        if not probe(target):
            print()
            print("[错误] 安装之后依赖仍然缺失。")
            print(f'        请执行：{target[0]} -m pip install -e "{INSTALL_SPEC}"')
            return 1
        print()
        print("       完成 —— Python 运行库与图形界面已就绪。")
        print("       所有依赖都在本目录内，未改动你系统里的 Python。")
        print()

    # ---- 第 4 步：权重与引擎。缺什么说什么，但**不阻断启动** ----
    runtime_ok, _, backends = module_report(target)
    states = scan_assets(ROOT)
    report = engine_report(states, runtime_ok=runtime_ok, backends=backends)
    ready = [s.label for s in report if s.ok]
    broken = [s for s in report if not s.ok]

    print("[4/5] 模型权重与引擎")
    if not broken:
        print(f"      全部可用：{' / '.join(ready)}")
    else:
        if ready:
            print(f"      可用：{' / '.join(ready)}")
        for status in broken:
            print(f"      不可用：{status.label} —— {status.reason}")
            fixes = [row for row in status.fix.splitlines() if row]
            for index, row in enumerate(fixes):
                print(f"              {'修复：' if index == 0 else '      '}{row}")
        print("      失去的只是对应引擎，界面与其他引擎照常可用；")
        print("      要看完整报告执行：start.bat doctor")
    for state in states:
        if state.undersized:
            print(
                f"      [注意] {state.asset.rel} 只有 {state.size} 字节，可能是桩模型或下载不完整"
            )
    print()

    print("[5/5] 正在启动...")
    if (target_log := log_target(ROOT)) is not None:
        print(f"      运行日志：{target_log}（出问题时把最新的 .log 一并反馈）")
    print()
    code = launch(target, args, ROOT)
    if code != 0:
        print()
        print("[错误] 启动失败。请把上面的报错信息一并反馈。")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
