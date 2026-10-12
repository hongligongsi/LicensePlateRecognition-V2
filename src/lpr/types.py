"""公共数据结构：所有识别引擎的统一输入输出契约。

无论是传统 CV 引擎还是深度学习引擎，都必须返回 ``PlateResult`` 列表，
这样界面层和 CLI 层就不需要知道底层用的是哪个引擎。

车牌的描述由三个**正交**维度构成：

* ``text``   —— 号码本身（如 ``京A12345``）
* ``color``  —— 底色（蓝/黄/绿/白/黑，外加危险品的黑底黄字）
* ``layout`` —— 版式（单行 / 双层）

颜色与版式必须分开表达：大型汽车号牌是"黄底双行"，小型汽车是"蓝底单行"，
而大型新能源车也有"黄绿双拼双行"的形态，仅用颜色或仅用版式都会丢信息。
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
    GREEN       绿色          新能源、农用、民航
    WHITE       白色          警用、军队、特种车辆
    YELLOW      黄色          大型汽车、教练车
    HAZMAT      —             危险品运输车辆（黑底黄字，需极性判定）
    ==========  ============  =====================================

    注意：**底色与版式是独立的两个维度**，见 :class:`PlateLayout`。
    同样是黄牌，小型教练车是单行、大型汽车后号牌是双层。
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


class PlateLayout(str, Enum):
    """车牌版式：单行 / 双层（双行）。

    版式与**底色是相互独立**的两个维度：大型汽车的后号牌与挂车号牌是
    "黄底 + 双行"，小型汽车是"蓝底 + 单行"，新能源是"绿底 + 单行"，
    大型新能源与部分特种车的绿牌也走双行。因此仅靠颜色无法判断版式，
    需要单独记录。

    ==========  ====================================================
    枚举值      说明
    ==========  ====================================================
    SINGLE      单行（横向一行字符）—— 绝大多数乘用车
    DOUBLE      双层 / 双行 —— 大型汽车后号牌、挂车号牌、双层绿牌
    UNKNOWN     未能判定（未检测到行分割或未展开双行形态时）
    ==========  ====================================================
    """

    SINGLE = "single"
    DOUBLE = "double"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return _LAYOUT_LABEL[self]


_LAYOUT_LABEL = {
    PlateLayout.SINGLE: "单行",
    PlateLayout.DOUBLE: "双层",
    PlateLayout.UNKNOWN: "未知版式",
}

#: 版式的规范显示名，供 ``--layout`` 参数与界面下拉使用
LAYOUT_NAMES = ("单行", "双层")

LAYOUT_ALIASES = {
    "单行": PlateLayout.SINGLE,
    "单": PlateLayout.SINGLE,
    "single": PlateLayout.SINGLE,
    "双层": PlateLayout.DOUBLE,
    "双行": PlateLayout.DOUBLE,
    "双": PlateLayout.DOUBLE,
    "double": PlateLayout.DOUBLE,
}


def parse_layout(name: str) -> PlateLayout:
    """把用户输入的版式名解析成枚举，不合法时抛 ``ValueError``。"""
    key = name.strip().lower()
    if key in LAYOUT_ALIASES:
        return LAYOUT_ALIASES[key]
    raise ValueError(f"未知车牌版式 {name!r}，可选: " + "、".join(LAYOUT_NAMES))


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
    #: 车牌版式（单行 / 双层）
    layout: PlateLayout = PlateLayout.UNKNOWN
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
            "layout": self.layout.value,
            "layout_label": self.layout.label,
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence": round(self.confidence, 4),
            "engine": self.engine,
            "char_confidences": [round(c, 4) for c in self.char_confidences],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlateResult":
        """从 :meth:`to_dict` 的结果还原。

        多进程批量识别要在父进程里重建结果对象，故需要这个反向入口。
        ``roi`` 不在 :meth:`to_dict` 的字段里（跨进程传图像开销过大），
        因此还原后 ``roi`` 恒为 ``None``。
        """
        color = PlateColor.UNKNOWN
        try:
            color = PlateColor(data.get("color") or "unknown")
        except ValueError:
            pass
        layout = PlateLayout.UNKNOWN
        try:
            layout = PlateLayout(data.get("layout") or "unknown")
        except ValueError:
            pass
        bbox = data.get("bbox")
        return cls(
            text=str(data.get("text") or ""),
            color=color,
            layout=layout,
            bbox=tuple(int(v) for v in bbox) if bbox else None,
            roi=None,
            confidence=float(data.get("confidence") or 0.0),
            engine=str(data.get("engine") or ""),
            char_confidences=[float(c) for c in (data.get("char_confidences") or [])],
            meta=dict(data.get("meta") or {}),
        )

    def __str__(self) -> str:
        if not self.ok:
            return "<未识别到车牌>"
        bbox = f" @{self.bbox}" if self.bbox else ""
        layout = "" if self.layout is PlateLayout.UNKNOWN else f" {self.layout.label}"
        return (
            f"{self.text} [{self.color.label}{layout}] "
            f"置信度={self.confidence:.2f}{bbox}"
        )
