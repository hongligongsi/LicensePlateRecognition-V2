"""CCPD 数据集标注解析与车牌区提取。

## 两种数据形态

**形态 A：原始 CCPD（推荐）** —— 标注编码在文件名里::

    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    │   │     │              │                        │                      │     └ 模糊度
    │   │     │              │                        │                      └ 亮度
    │   │     │              │                        └ 车牌号码（字符下标，_ 分隔）
    │   │     │              └ 车牌四角点
    │   │     └ bbox 左上/右下
    │   └ 水平/垂直倾角
    └ 车牌面积占比

**形态 B：ModelScope 镜像** —— 文件名被重写成 ``sample_N.jpg``，标注丢失。
此时只能做**无监督**评估（靠定位算法自身切出车牌，再用一致性指标打分），
见 :func:`crop_by_detection`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: CCPD 字符表（下标即文件名中号码字段的取值）。
#: 前 31 个是省份简称，随后是字母表，最后是数字与 4 个特殊尾字。
CCPD_CHARS = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "学",
    "警",
    "港",
    "澳",
]


@dataclass(frozen=True)
class CcpdAnnotation:
    """单张 CCPD 图片的标注（形态 A）。"""

    path: Path
    text: str
    bbox: tuple[int, int, int, int]
    corners: list[tuple[int, int]]
    tilt: tuple[int, int]
    area_ratio: float
    brightness: int
    blurriness: int

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        """按四角点求外接矩形，比 bbox 更贴合真实车牌。"""
        xs = [p[0] for p in self.corners]
        ys = [p[1] for p in self.corners]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def is_green(self) -> bool:
        """新能源牌（8 位）在 CCPD 中约占少数。"""
        return len(self.text) == 8


def _decode_plate(field: str) -> str:
    """把 ``0_0_22_27_27_33_16`` 这样的下标串还原成车牌号。"""
    out = []
    for token in field.split("_"):
        try:
            idx = int(token)
        except ValueError:
            continue
        if 0 <= idx < len(CCPD_CHARS):
            out.append(CCPD_CHARS[idx])
    return "".join(out)


def has_labels(directory: str | Path) -> bool:
    """判断目录里是否含原始 CCPD 标注（文件名形态而非 ``sample_N``）。"""
    for p in Path(directory).glob("*.jpg"):
        if p.stem.startswith("sample_") and p.stem[7:].isdigit():
            continue
        if len(p.stem.split("-")) >= 7:
            return True
    return False


def parse(name: str | Path) -> CcpdAnnotation | None:
    """从文件名解析标注，格式不符返回 ``None``。"""
    path = Path(name)
    parts = path.stem.split("-")
    if len(parts) < 7:
        return None

    try:
        area_ratio = float(parts[0]) / 100.0
        tilt_part = parts[1].split("_")
        tilt = (int(tilt_part[0]), int(tilt_part[1])) if len(tilt_part) >= 2 else (0, 0)

        lt, rb = parts[2].split("_")
        x1, y1 = (int(v) for v in lt.split("&"))
        x2, y2 = (int(v) for v in rb.split("&"))

        corners = []
        for pt in parts[3].split("_"):
            cx, cy = pt.split("&")
            corners.append((int(cx), int(cy)))

        text = _decode_plate(parts[4])
        brightness = int(parts[5])
        blurriness = int(parts[6])
    except (ValueError, IndexError):
        return None

    if not text or not corners:
        return None
    return CcpdAnnotation(
        path, text, (x1, y1, x2, y2), corners, tilt, area_ratio, brightness, blurriness
    )


def load_dir(directory: str | Path) -> list[CcpdAnnotation]:
    """加载目录下所有**带标注**的 CCPD 图片。"""
    out = []
    for p in sorted(Path(directory).glob("*.jpg")):
        ann = parse(p)
        if ann is not None:
            out.append(ann)
    return out


def crop_by_detection(img, ann_or_none=None, pad: float = 0.0):
    """用标注四角点切出车牌区域（形态 A 专用）。

    ``pad`` 为相对外扩比例，正数向外留白、负数向内收缩。
    """

    if ann_or_none is None:
        return img
    x1, y1, x2, y2 = ann_or_none.crop_box
    h, w = img.shape[:2]
    if pad:
        dx = int((x2 - x1) * pad)
        dy = int((y2 - y1) * pad)
        x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]
