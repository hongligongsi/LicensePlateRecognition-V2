"""图像读取等通用工具。

原项目的 ``imreadex`` 用 ``np.fromfile + cv2.imdecode`` 是为了兼容中文路径，
这一点是对的，保留；这里额外补上：
  * 读取失败时给出明确异常而不是返回 None；
  * EXIF 方向自动纠正（手机拍摄的照片经常带旋转信息，原版会横过来）；
  * 绘制车牌框、结果叠加等界面/CLI 共用的辅助函数。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..types import COLOR_SWATCH, PlateResult


class ImageReadError(RuntimeError):
    """图片无法读取。"""


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图片，正确处理中文/Unicode 路径。

    与 ``cv2.imread`` 不同，这里不会因为路径含中文而静默返回 ``None``。
    """
    path = Path(path)
    if not path.exists():
        raise ImageReadError(f"文件不存在: {path}")

    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise ImageReadError(f"文件为空: {path}")

    img = cv2.imdecode(buf, flags)
    if img is None:
        raise ImageReadError(f"无法解码为图像（可能不是图片或格式不支持）: {path}")

    return _apply_exif_orientation(img, path)


def _apply_exif_orientation(img: np.ndarray, path: Path) -> np.ndarray:
    """根据 EXIF 旋转标记把图片摆正。"""
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover - Pillow 是可选依赖
        return img

    if path.suffix.lower() not in (".jpg", ".jpeg", ".tif", ".tiff"):
        return img

    try:
        with Image.open(path) as pil_img:
            transposed = ImageOps.exif_transpose(pil_img)
            if transposed is pil_img:
                return img
            return cv2.cvtColor(np.array(transposed.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        # EXIF 解析失败不影响主流程
        return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    """写出图片，同样支持中文路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, buf = cv2.imencode(suffix, img)
    if not ok:
        raise ImageReadError(f"图片编码失败: {path}")
    buf.tofile(str(path))


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """把长边限制到 ``max_side``，返回（新图, 缩放系数）。

    缩放系数用于把坐标映射回原图。
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0

    scale = max_side / longest
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4), scale


def ensure_odd(value: int, minimum: int = 1) -> int:
    """把值修正为不小于 minimum 的奇数，用于形态学核。"""
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def ensure_bgr(img: np.ndarray | None, min_side: int = 2) -> np.ndarray | None:
    """把任意形态的输入图像归一化为**三通道 uint8 BGR**，不可用则返回 ``None``。

    图像处理管线里的每个入口（版式测量、颜色判定、双层切分……）都可能
    收到非常规输入：灰度图、带 alpha 的四通道图、float32/float64、
    含 NaN 的数组、甚至 ``None``。这些输入直接喂给 OpenCV 会抛出五花八门
    的异常。与其在每个函数里各自防护，不如在入口统一归一化：

    * ``None`` / 维度异常 / 过小（单边 < ``min_side``）→ ``None``
    * 灰度（2 维或第 3 维为 1）→ 转三通道
    * 四通道（BGRA）→ 去 alpha
    * 非整型像素（float 等）→ NaN/inf 置 0 后裁剪到 0~255 转 uint8
    """
    if img is None or not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
        return None
    h, w = img.shape[:2]
    if h < min_side or w < min_side:
        return None

    arr = img
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.shape[2] != 3:
        return None

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 255.0)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return arr


def draw_result(img: np.ndarray, results: list[PlateResult]) -> np.ndarray:
    """在原图上画出所有识别到的车牌框和文字，返回新图。

    中文标签用 Pillow 绘制（``cv2.putText`` 不支持中文，会画出 ???）。
    """
    canvas = img.copy()
    boxes_to_label: list[tuple[tuple[int, int, int, int], PlateColor, str]] = []

    for res in results:
        if not res.bbox:
            continue
        x, y, w, h = res.bbox
        color = COLOR_SWATCH.get(res.color, (0, 200, 255))
        thickness = max(2, round(max(canvas.shape[:2]) / 500))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        label = res.text or res.color.label
        boxes_to_label.append(((x, y, w, h), res.color, label))

    if not boxes_to_label:
        return canvas

    _draw_chinese_labels(canvas, boxes_to_label)
    return canvas


def _pillow_font(size: int):
    """加载一个支持中文的字体，找不到就用默认（可能仍显示方块）。"""
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_chinese_labels(
    canvas: np.ndarray,
    items: list[tuple[tuple[int, int, int, int], "PlateColor", str]],  # noqa: F821
) -> None:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")
    font_size = max(16, round(max(canvas.shape[:2]) / 55))
    font = _pillow_font(font_size)

    for (x, y, w, _h), color, label in items:
        r, g, b = color.value.rgb if hasattr(color.value, "rgb") else _swatch_rgb(color)
        pad = max(2, font_size // 6)
        left, top = x, max(0, y - font_size - pad * 3)
        bbox = draw.textbbox((left + pad, top + pad), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 亮色底用黑字，深色底用白字
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        text_fill = (0, 0, 0, 255) if lum > 140 else (255, 255, 255, 255)

        draw.rectangle(
            [left, top, left + text_w + pad * 2, top + text_h + pad * 2],
            fill=(r, g, b, 235),
        )
        draw.text((left + pad, top + pad), label, font=font, fill=text_fill)

    canvas[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _swatch_rgb(color: "PlateColor") -> tuple[int, int, int]:  # noqa: F821
    from ..types import COLOR_SWATCH

    b, g, r = COLOR_SWATCH.get(color, (160, 160, 160))
    return r, g, b


def to_display_rgb(img: np.ndarray) -> np.ndarray:
    """BGR → RGB，供 Qt/Pillow 显示。"""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


__all__ = [
    "imread",
    "imwrite",
    "resize_max_side",
    "ensure_odd",
    "ensure_bgr",
    "draw_result",
    "to_display_rgb",
    "ImageReadError",
]
