"""``python -m lpr`` 入口 —— 不安装 console script 也能用。

* ``python -m lpr``                            → 启动图形界面（等价 ``lpr gui``）
* ``python -m lpr predict car.jpg --engine deep`` → 完整命令行（子命令原样透传）

无参数时默认进界面，这是"刚 clone 下来就想看效果"的最短路径；
若想看命令行用法，用 ``python -m lpr.cli --help``（那里无参数会打印帮助而不是开界面）。
"""

from __future__ import annotations

import sys

from .cli import main

_argv = sys.argv[1:]

#: 无参数 → 直接开图形界面；带参数 → 原样交给 CLI 子命令解析
if not _argv:
    _argv = ["gui"]

sys.exit(main(_argv))
