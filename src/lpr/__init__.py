"""车牌识别 v2 —— 主包。

推荐用法（高层 API）::

    from lpr import Recognizer
    rec = Recognizer(engine="auto")     # auto / cv / deep / cascade
    outcome = rec.recognize("car.jpg")
    print(outcome.best.text)

底层用法（直接操作引擎）::

    from lpr import get_engine
    engine = get_engine("cv")
    results = engine.predict("car.jpg")
"""

from __future__ import annotations

from .api import BatchOutcome, RecognizeOutcome, Recognizer
from .config import Config, SecurityConfig
from .engines import (
    CVEngine,
    DeepEngine,
    available_engines,
    describe_engines,
    get_engine,
    register_engine,
)
from .plate_rules import correct, is_valid, plate_score
from .plate_rules import describe as describe_plate
from .tracker import MultiPlateTracker, merge_same_image
from .types import COLOR_NAMES, PlateColor, PlateResult, parse_color
from .utils.logging_setup import setup_logging
from .utils.security import (
    Limits,
    PrivacyPolicy,
    SecurityContext,
    SecurityError,
    mask_plate,
    verify_model,
)

#: 版本号唯一来源。``cli.VERSION``、界面标题、状态栏都从这里取，
#: 不再各自维护字面量（v2.3.9 之前这里停在 2.3.1，与 pyproject 各说各话）。
__version__ = "2.3.9"

__all__ = [
    # 高层 API
    "Recognizer",
    "RecognizeOutcome",
    "BatchOutcome",
    # 配置
    "Config",
    "SecurityConfig",
    "Limits",
    "PrivacyPolicy",
    "SecurityContext",
    "SecurityError",
    # 引擎
    "CVEngine",
    "DeepEngine",
    "get_engine",
    "register_engine",
    "available_engines",
    "describe_engines",
    # 数据
    "PlateColor",
    "PlateResult",
    "COLOR_NAMES",
    "parse_color",
    # 规则与跟踪
    "correct",
    "is_valid",
    "plate_score",
    "describe_plate",
    "MultiPlateTracker",
    "merge_same_image",
    # 安全工具
    "mask_plate",
    "verify_model",
    "setup_logging",
    "__version__",
]
