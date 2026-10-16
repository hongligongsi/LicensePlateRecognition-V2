"""识别引擎包。

导入本包会触发引擎注册，使 ``lpr.engines.get_engine("cv")`` 可用。
"""

from __future__ import annotations

from .base import BaseEngine, EngineUnavailable

# 导入即注册内置引擎（放在 registry 之后，避免循环导入）
from .cv_engine import CVEngine
from .deep_engine import DeepEngine
from .fusion_engine import FusionEngine
from .registry import (
    ALIAS_ENGINES,
    INFERENCE_LOCK,
    available_engines,
    clear_instances,
    create_engine,
    describe_engine_choices,
    describe_engines,
    get_engine,
    register_engine,
)
from .yolo_engine import YoloEngine

register_engine(CVEngine)
register_engine(DeepEngine)
register_engine(YoloEngine)
register_engine(FusionEngine)

__all__ = [
    "ALIAS_ENGINES",
    "INFERENCE_LOCK",
    "BaseEngine",
    "CVEngine",
    "DeepEngine",
    "EngineUnavailable",
    "FusionEngine",
    "YoloEngine",
    "available_engines",
    "clear_instances",
    "create_engine",
    "describe_engine_choices",
    "describe_engines",
    "get_engine",
    "register_engine",
]
