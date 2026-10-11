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
    """车牌底色。

    支持五种常规底色 + 危险品运输车牌：

    ==========  ============  =====================================
    枚举值      行驶证颜色     说明
    ==========  ============  =====================================
    BLACK       黑色          使领馆、境外车辆
    BLUE        蓝色          小型汽车（最常见）
    GREEN       绿色          新能源、农用
    WHITE       白色          警用、军队、特种车辆
    YELLOW      黄色          大型汽车、教练车
    HAZMAT      —             危险品运输车辆（黑底黄字，需极性判定）
    ==========  ============  =====================================
    """

    BLACK = "black"
    BLUE = "blue"
    GREEN = "green"
    WHITE = "white"
    YELLOW = "yellow"
    HAZMAT = "hazmat"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return _COLOR_LABEL[self]


#: 五种常规底色的规范显示名（与行驶证用词一致）
COLOR_NAMES = ("黑色", "蓝色", "绿色", "白色", "黄色")

#: 五种常规底色对应的中文规范名，供 ``--color`` 参数与界面下拉使用
COLOR_ALIASES = {
    "黑": PlateColor.BLACK,
    "黑色": PlateColor.BLACK,
    "black": PlateColor.BLACK,
    "蓝": PlateColor.BLUE,
    "蓝色": PlateColor.BLUE,
    "blue": PlateColor.BLUE,
    "绿": PlateColor.GREEN,
    "绿色": PlateColor.GREEN,
    "green": PlateColor.GREEN,
    "白": PlateColor.WHITE,
    "白色": PlateColor.WHITE,
    "white": PlateColor.WHITE,
    "黄": PlateColor.YELLOW,
    "黄色": PlateColor.YELLOW,
    "yellow": PlateColor.YELLOW,
    "危险品": PlateColor.HAZMAT,
    "危险品运输": PlateColor.HAZMAT,
    "hazmat": PlateColor.HAZMAT,
}


def parse_color(name: str) -> PlateColor:
    """把用户输入的颜色名解析成枚举，不合法时抛 ``ValueError``。"""
    key = name.strip().lower()
    if key in COLOR_ALIASES:
        return COLOR_ALIASES[key]
    raise ValueError(
        f"未知车牌颜色 {name!r}，可选: "
        + "、".join(COLOR_NAMES + ("危险品",))
    )


_COLOR_LABEL = {
    PlateColor.BLACK: "黑牌",
    PlateColor.BLUE: "蓝牌",
    PlateColor.GREEN: "绿牌（新能源）",
    PlateColor.WHITE: "白牌",
    PlateColor.YELLOW: "黄牌",
    PlateColor.HAZMAT: "危险品运输牌（黑底黄字）",
    PlateColor.UNKNOWN: "未知",
}

#: 界面中用于标出车牌颜色的色块（BGR）
COLOR_SWATCH = {
    PlateColor.BLACK: (60, 60, 60),
    PlateColor.BLUE: (255, 102, 102),
    PlateColor.GREEN: (85, 255, 85),
    PlateColor.WHITE: (255, 255, 255),
    PlateColor.YELLOW: (0, 255, 255),
    PlateColor.HAZMAT: (0, 200, 255),
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
