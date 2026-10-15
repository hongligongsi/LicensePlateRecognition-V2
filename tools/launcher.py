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
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

#: 与 pyproject.toml / cli.py / config.py 保持一致
VERSION = "2.5.1"
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
# 基础动作
# ────────────────────────────────────────────────────────────────
def probe(interp: Sequence[str], timeout: int = 120) -> bool:
    """目标解释器是否已具备全部运行依赖（静默，只看返回码）。"""
    try:
        proc = subprocess.run(
            [*interp, "-c", PROBE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


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


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

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
    print("[1/4] 解释器正常")
    print("       ", sys.executable)

    target, action = choose_target(interp, ROOT, no_venv=bool(os.environ.get("LPR_NO_VENV")))

    if action == "reuse":
        print("[2/4] 依赖齐全")
        print("[3/4] 无需安装")
    else:
        print("[2/4] 依赖缺失，正在准备运行环境...")
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

        print(f"[3/4] 正在安装到 {target[0]}（仅首次，需要联网）...")
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
        print("       完成 —— CV 引擎、深度引擎、图形界面均已就绪。")
        print("       所有依赖都在本目录内，未改动你系统里的 Python。")
        print()

    print("[4/4] 正在启动...")
    print()
    code = launch(target, args, ROOT)
    if code != 0:
        print()
        print("[错误] 启动失败。请把上面的报错信息一并反馈。")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
