"""日志配置。

原项目用 ``print`` 输出一堆调试信息（``len(contours)``、``peak less 1:`` 等），
在界面上会刷屏。这里统一走 logging，并且提供一个内存 handler 方便界面显示。
"""

from __future__ import annotations

import logging
import sys
from collections import deque

_LOGGER_NAME = "lpr"
_configured = False


class RingBufferHandler(logging.Handler):
    """把最近 N 条日志留在内存里，供界面"运行日志"面板展示。"""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.records: deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:  # pragma: no cover - 日志本身不应抛异常
            pass

    def dump(self) -> list[str]:
        return list(self.records)


#: 全局共享的内存日志缓冲，界面可直接读取
memory_handler = RingBufferHandler()


def setup_logging(level: str | int = "INFO") -> logging.Logger:
    """初始化 lpr 命名空间的日志，重复调用不会叠加 handler。"""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    if not _configured:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        memory_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(stream)
        logger.addHandler(memory_handler)
        logger.propagate = False
        _configured = True

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """取得子 logger，例如 ``get_logger("engines.cv")``。"""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


__all__ = ["setup_logging", "get_logger", "memory_handler", "RingBufferHandler"]
