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
from .utils.logging_setup import current_log_file, log_file_path, setup_logging
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
__version__ = "2.7.3"

__all__ = [
    "COLOR_NAMES",
    "BatchOutcome",
    # 引擎
    "CVEngine",
    # 配置
    "Config",
    "DeepEngine",
    "Limits",
    "MultiPlateTracker",
    # 数据
    "PlateColor",
    "PlateResult",
    "PrivacyPolicy",
    "RecognizeOutcome",
    # 高层 API
    "Recognizer",
    "SecurityConfig",
    "SecurityContext",
    "SecurityError",
    "__version__",
    "available_engines",
    # 规则与跟踪
    "correct",
    "current_log_file",
    "describe_engines",
    "describe_plate",
    "get_engine",
    "is_valid",
    # 日志
    "log_file_path",
    # 安全工具
    "mask_plate",
    "merge_same_image",
    "parse_color",
    "plate_score",
    "register_engine",
    "setup_logging",
    "verify_model",
]
