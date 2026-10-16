"""日志配置。

原项目用 ``print`` 输出一堆调试信息（``len(contours)``、``peak less 1:`` 等），
在界面上会刷屏。这里统一走 logging。

日志有**三个互不替代的出口**：

* **控制台**（stderr）：给人当场看，默认开；
* **文件**（``logs/lpr-YYYYMMDD.log``，按大小轮转）：给事后排查用，默认开。
  v2.7.2 之前只有控制台 —— 窗口一关日志就没了，用户报障时手上没有任何证据；
* **内存环形缓冲**（:data:`memory_handler`）：给界面的「运行日志」面板读。

写文件失败（目录只读、磁盘满、被杀软占用）**只降级为"不写文件"**，
绝不抛异常：真正需要日志的时刻，恰恰就是环境已经不正常的时候。
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "lpr"

#: 默认日志目录（相对**当前工作目录**，不是包安装目录）
DEFAULT_LOG_DIR = "logs"
#: 单个日志文件默认上限（MB），超出后轮转
DEFAULT_MAX_MB = 5
#: 默认保留的历史日志份数
DEFAULT_BACKUPS = 5
#: 认可的级别名（不含 ``logging`` 里那些整型常量，避免 ``getattr`` 撞上）
LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LEVEL_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}
#: 文件名里的日期戳
_STAMP_FORMAT = "%Y%m%d"

_configured = False
_file_handler: RotatingFileHandler | None = None
_log_file: Path | None = None


class RingBufferHandler(logging.Handler):
    """把最近 N 条日志留在内存里，供界面"运行日志"面板展示。"""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.records: deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        # pragma: no cover - 日志本身不应抛异常
        with contextlib.suppress(Exception):
            self.records.append(self.format(record))

    def dump(self) -> list[str]:
        return list(self.records)

    def clear(self) -> None:
        self.records.clear()


#: 全局共享的内存日志缓冲，界面可直接读取
memory_handler = RingBufferHandler()


def log_file_path(
    log_dir: str | Path = DEFAULT_LOG_DIR,
    *,
    when: datetime | None = None,
    tag: str = "",
) -> Path:
    """算出日志文件路径：``<log_dir>/lpr-<YYYYMMDD><tag>.log``。

    ``tag`` 是给**多进程**用的：并行批量的每个子进程通过 ``tag=f"-{pid}"``
    写自己的文件。这不是洁癖 —— ``RotatingFileHandler`` 轮转时要重命名文件，
    而 Windows 上重命名一个仍被别的进程打开的文件会直接抛 ``PermissionError``，
    多个子进程共享一个日志文件时会在跑量大的批次里随机炸掉。
    """
    stamp = (when or datetime.now()).strftime(_STAMP_FORMAT)
    return Path(log_dir) / f"lpr-{stamp}{tag}.log"


def current_log_file() -> Path | None:
    """本进程正在写的日志文件；未开启（或打开失败）时为 ``None``。"""
    return _log_file


def _coerce_level(level: str | int) -> tuple[int, bool]:
    """把级别转成 ``logging`` 的整型值，返回 ``(值, 是否是未知名字)``。"""
    if isinstance(level, int):  # bool 也走这里，无妨
        return level, False
    name = str(level).strip().upper()
    name = _LEVEL_ALIASES.get(name, name)
    if name in LEVEL_CHOICES:
        return getattr(logging, name), False
    return logging.INFO, True


def _open_log_file(
    log_dir: str | Path,
    max_mb: int,
    backups: int,
    fmt: logging.Formatter,
    tag: str,
) -> tuple[RotatingFileHandler | None, Path | None]:
    """开会话的日志文件；打不开就返回 ``(None, None)`` 并把原因打到 stderr。"""
    path = log_file_path(log_dir, tag=tag)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=max(1, int(max_mb)) * 1024 * 1024,
            backupCount=max(0, int(backups)),
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, ValueError) as exc:
        print(f"[警告] 无法写入日志文件 {path}：{exc}", file=sys.stderr)
        return None, None
    handler.setFormatter(fmt)
    return handler, path


def setup_logging(
    level: str | int = "INFO",
    *,
    log_to_file: bool = True,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    max_mb: int = DEFAULT_MAX_MB,
    backups: int = DEFAULT_BACKUPS,
    console: bool = True,
    tag: str = "",
    force: bool = False,
) -> logging.Logger:
    """初始化 ``lpr`` 命名空间的日志，重复调用不会叠加 handler。

    参数
    ----
    level:
        级别名或 ``logging`` 整型值。名字不认得时按 ``INFO`` 处理并**发一条
        警告**（静默降级会让人以为设错了参数没生效）。
    log_to_file / log_dir / max_mb / backups:
        文件日志开关、目录、单文件上限（MB）与保留份数。
    console:
        是否同时打到 stderr（批量脚本想把日志留给自己时关掉）。
    tag:
        文件名后缀，多进程子进程用 ``f"-{os.getpid()}"`` 区分。
    force:
        已经配置过时**重建** handler（用于运行期换级别/换文件）。
        默认 ``False``，即幂等。
    """
    global _configured, _file_handler, _log_file

    logger = logging.getLogger(_LOGGER_NAME)
    if _configured and not force:
        return logger
    if _configured:
        close_logging()

    level_value, unknown = _coerce_level(level)
    logger.setLevel(level_value)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        logger.addHandler(stream)

    memory_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(memory_handler)

    # ``log_dir`` 为空串等价于关掉文件日志：允许 ``log_dir=""`` 这种惰性写法
    if log_to_file and str(log_dir).strip():
        _file_handler, _log_file = _open_log_file(log_dir, max_mb, backups, fmt, tag)
        if _file_handler is not None:
            logger.addHandler(_file_handler)

    # 关掉向上传播：根 logger 上没有我们的 handler，传播出去只会触发
    # logging 的 lastResort（WARNING 以上打到 stderr，INFO 直接丢）
    logger.propagate = False
    _configured = True

    if unknown:
        logger.warning(
            "未知日志级别 %r，已按 INFO 处理（可选：%s）", level, " / ".join(LEVEL_CHOICES)
        )
    if _log_file is not None:
        # 第一行就写清楚日志在哪 —— 用户问"日志在哪"是最高频的追问。
        # 用 ``absolute()`` 而不是 ``resolve()``：后者在 Windows 上会走
        # realpath，可能返回 ``C:\Users\HONGLI~1\...`` 这种 8.3 短名，难认。
        logger.info("运行日志 → %s", _log_file.absolute())
    return logger


def close_logging() -> None:
    """摘掉并关闭本模块装上的 handler（测试收尾、运行期重建时用）。

    注意：只关文件句柄，**不**关 ``sys.stderr`` —— StreamHandler 的
    ``close()`` 本来就不碰流对象，这里额外显式跳过内存 handler，
    因为它是模块级单例，还有下一次要用。
    """
    global _configured, _file_handler, _log_file

    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        if handler is memory_handler:
            continue
        with contextlib.suppress(Exception):
            handler.close()

    _file_handler = None
    _log_file = None
    _configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    """取得子 logger，例如 ``get_logger("engines.cv")``。"""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


__all__ = [
    "DEFAULT_BACKUPS",
    "DEFAULT_LOG_DIR",
    "DEFAULT_MAX_MB",
    "LEVEL_CHOICES",
    "RingBufferHandler",
    "close_logging",
    "current_log_file",
    "get_logger",
    "log_file_path",
    "memory_handler",
    "setup_logging",
]
