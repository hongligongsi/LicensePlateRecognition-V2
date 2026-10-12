"""识别引擎包。

导入本包会触发引擎注册，使 ``lpr.engines.get_engine("cv")`` 可用。
"""

from __future__ import annotations

from .base import BaseEngine, EngineUnavailable

# 导入即注册内置引擎（放在 registry 之后，避免循环导入）
from .cv_engine import CVEngine
from .deep_engine import DeepEngine
from .registry import (
    INFERENCE_LOCK,
    available_engines,
    clear_instances,
    create_engine,
    describe_engines,
    get_engine,
    register_engine,
)

register_engine(CVEngine)
register_engine(DeepEngine)

__all__ = [
    "INFERENCE_LOCK",
    "BaseEngine",
    "CVEngine",
    "DeepEngine",
    "EngineUnavailable",
    "available_engines",
    "clear_instances",
    "create_engine",
    "describe_engines",
    "get_engine",
    "register_engine",
]
