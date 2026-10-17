"""运行环境指纹：一行日志回答"这次跑的是什么"。

为什么需要它：排查"同一张图昨天能认、今天认不出"时，旧日志里只有引擎就绪
那一行（``后端=rapidocr | 定位引导=启用 …``），**分不清是代码变了、配置变了
还是环境变了** —— 三者的修法完全不同。加一行指纹后，两次运行的日志直接
可比：

    环境指纹: lpr 2.8.1 | Python 3.13.12 (Windows) | NumPy 2.1.3 | OpenCV 4.11.0
              | 引擎默认=auto | 配置指纹=3f9a1c02

设计上的两个取舍：

* **配置指纹用哈希而不是全文**：日志要能长期留存，把整份 config 打进去既
  冗长又可能带出路径等隐私信息。8 位十六进制足够回答"两次是不是同一份配置"。
  想还原具体值随时可以 ``lpr config show``。
* **cv2 / numpy 取已加载模块，不主动 import**：环境指纹不该为了打印版本号
  把 OpenCV 拉起来（CLI 的 ``check``/``config`` 等子命令根本用不到它）。
  没加载就如实写"未加载"。
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from typing import Any

from ..config import Config
from .logging_setup import get_logger

logger = get_logger("utils.environment")


def config_fingerprint(config: Config) -> str:
    """把**影响识别结果**的那部分配置压成 8 位指纹。

    只取 ``engine`` 段（含 ``locate`` 与 ``fusion`` 的全部阈值）：``ui`` /
    ``logging`` / ``security`` 改了不会让同一张图读出不同的车牌，混进来只会
    让"配置指纹变了"变成噪音。
    """
    payload: dict[str, Any] = asdict(config.engine)
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def _module_version(name: str) -> str:
    """取**已加载**模块的版本；没加载就说没加载，不替它 import。"""
    mod = sys.modules.get(name)
    if mod is None:
        return "未加载"
    return str(getattr(mod, "__version__", "未知"))


def environment_line(config: Config) -> str:
    """构造环境指纹（单行，不含换行 —— 日志按行 grep 才方便）。"""
    from .. import __version__

    return (
        f"环境指纹: lpr {__version__}"
        f" | Python {platform.python_version()} ({platform.system() or '未知'})"
        f" | NumPy {_module_version('numpy')}"
        f" | OpenCV {_module_version('cv2')}"
        f" | 引擎默认={config.engine.default}"
        f" | 配置指纹={config_fingerprint(config)}"
    )


def log_environment(config: Config) -> str:
    """打一行环境指纹并返回它（返回值供测试断言，不重复格式化）。"""
    line = environment_line(config)
    logger.info("%s", line)
    return line


__all__ = ["config_fingerprint", "environment_line", "log_environment"]
