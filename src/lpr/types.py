"""公共数据结构：所有识别引擎的统一输入输出契约。

无论是传统 CV 引擎还是深度学习引擎，都必须返回 ``PlateResult`` 列表，
这样界面层和 CLI 层就不需要知道底层用的是哪个引擎。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class PlateColor(str, Enum):
    """车牌底色。原项目只用字符串，这里改为枚举避免拼写错误。"""

    BLUE = "blue"
    YELLOW = "yellow"
    GREEN = "green"
    WHITE = "white"
    BLACK = "black"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return _COLOR_LABEL[self]


_COLOR_LABEL = {
    PlateColor.BLUE: "蓝牌",
    PlateColor.YELLOW: "黄牌",
    PlateColor.GREEN: "绿牌（新能源）",
    PlateColor.WHITE: "白牌",
    PlateColor.BLACK: "黑牌",
    PlateColor.UNKNOWN: "未知",
}

#: 界面中用于标出车牌颜色的色块（BGR）
COLOR_SWATCH = {
    PlateColor.BLUE: (255, 102, 102),
    PlateColor.YELLOW: (0, 255, 255),
    PlateColor.GREEN: (85, 255, 85),
    PlateColor.WHITE: (255, 255, 255),
    PlateColor.BLACK: (60, 60, 60),
    PlateColor.UNKNOWN: (160, 160, 160),
}


@dataclass
class PlateResult:
    """一次识别得到的单个车牌结果。"""

    #: 识别出的车牌号码，例如 ``"京A12345"``；识别失败时为空字符串
    text: str = ""
    #: 车牌底色
    color: PlateColor = PlateColor.UNKNOWN
    #: 车牌在原图中的位置 ``(x, y, w, h)``，未知时为 ``None``
    bbox: tuple[int, int, int, int] | None = None
    #: 车牌区域图像（BGR 或灰度），用于界面展示
    roi: np.ndarray | None = None
    #: 整体置信度 0~1；传统 CV 引擎给出的是字符级平均置信度
    confidence: float = 0.0
    #: 产生该结果的引擎名称
    engine: str = ""
    #: 各字符的识别置信度，便于定位是哪个字识别错了
    char_confidences: list[float] = field(default_factory=list)
    #: 引擎相关的附加信息（耗时、候选数等），仅用于调试
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """是否成功识别出号码。"""
        return bool(self.text)

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典（不含图像）。"""
        return {
            "text": self.text,
            "color": self.color.value,
            "color_label": self.color.label,
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence": round(self.confidence, 4),
            "engine": self.engine,
            "char_confidences": [round(c, 4) for c in self.char_confidences],
            "meta": self.meta,
        }

    def __str__(self) -> str:
        if not self.ok:
            return "<未识别到车牌>"
        bbox = f" @{self.bbox}" if self.bbox else ""
        return f"{self.text} [{self.color.label}] 置信度={self.confidence:.2f}{bbox}"
