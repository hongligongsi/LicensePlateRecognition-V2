"""车牌识别 v2 —— 主包。

用法：
    from lpr import get_engine
    engine = get_engine("cv")   # 或 "deep"
    results = engine.predict("car.jpg")
"""

from __future__ import annotations

from .config import Config
from .engines import (
    CVEngine,
    DeepEngine,
    available_engines,
    describe_engines,
    get_engine,
    register_engine,
)
from .types import PlateColor, PlateResult
from .utils.logging_setup import setup_logging

__version__ = "2.0.0"

__all__ = [
    "Config",
    "PlateColor",
    "PlateResult",
    "CVEngine",
    "DeepEngine",
    "get_engine",
    "register_engine",
    "available_engines",
    "describe_engines",
    "setup_logging",
    "__version__",
]
