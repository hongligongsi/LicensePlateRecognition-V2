#!/usr/bin/env python
"""生成六种底色的合成车牌样张，用于验证底色与危险品判定。

`lpr predict --color` 的效果无法用现有 samples/ 完整演示（那里只有黄牌和绿牌），
这个脚本合成一套标准样张，方便人工目视核对与回归测试。

用法::

    python tools/make_color_samples.py                  # 输出到 out/color_samples/
    python tools/make_color_samples.py --out my_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lpr.config import LocateConfig  # noqa: E402
from lpr.engines.locate import detect_color  # noqa: E402
from lpr.types import PlateColor  # noqa: E402

#: (文件名, 号码, 底色 BGR, 字色 BGR, 期望判定)
SAMPLES = [
    ("01_蓝牌.jpg", "京A12345", (196, 78, 20), (255, 255, 255), PlateColor.BLUE),
    ("02_黄牌.jpg", "京B12345", (0, 208, 245), (20, 20, 20), PlateColor.YELLOW),
    ("03_绿牌.jpg", "京AD77972", (86, 198, 40), (15, 15, 15), PlateColor.GREEN),
    ("04_白牌.jpg", "京A1234警", (238, 238, 238), (20, 20, 20), PlateColor.WHITE),
    ("05_黑牌.jpg", "京A12345", (24, 24, 24), (240, 240, 240), PlateColor.BLACK),
    # 危险品：黑底黄字（与普通黄牌底色相反）
    ("06_危险品.jpg", "京A1234危", (18, 18, 18), (0, 215, 255), PlateColor.HAZMAT),
]

_W, _H = 440, 140


def _render(bg: tuple[int, int, int], fg: tuple[int, int, int], text: str) -> np.ndarray:
    """画一块车牌：底色 + 描边 + 字符 + 轻微噪声。

    刻意加噪声与描边，是为了让 CV 定位器（依赖"密集边缘"）能真正找到它 ——
    纯色平铺图在形态学开运算后与原图相减得不到有效边缘，定位会失败。
    """
    img = np.full((_H, _W, 3), bg, np.uint8)

    # 外边框（车牌边缘通常有一圈浅色或描边）
    cv2.rectangle(img, (2, 2), (_W - 3, _H - 3), (235, 235, 235), 3)

    # 号码：省份简称稍大，其余字符等宽排布
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text[:1], (18, 100), font, 1.6, fg, 4, cv2.LINE_AA)
    cv2.putText(img, text[1:], (78, 100), font, 1.7, fg, 4, cv2.LINE_AA)

    # 两字符之间的圆点分隔符（中文车牌习惯）
    if len(text) > 2:
        cv2.circle(img, (70, 92), 5, fg, -1, cv2.LINE_AA)

    # 轻微高斯噪声：模拟真实拍摄，同时给定位器提供可用的边缘
    noise = np.random.default_rng(seed=abs(hash(text)) % (2**31)).normal(
        0, 6.0, img.shape
    )
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成六种底色的合成车牌样张")
    parser.add_argument("--out", default="out/color_samples", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = LocateConfig()
    ok = 0

    print(f"{'文件':<20}{'判定结果':<26}{'期望':<26}{'结果'}")
    print("-" * 84)

    for name, text, bg, fg, expect in SAMPLES:
        img = _render(bg, fg, text)
        path = out_dir / name
        cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(path))

        got, ratio = detect_color(img, cfg)
        good = got is expect
        ok += good
        mark = "OK" if good else "FAIL"
        print(f"{name:<18}{got.label:<24}{expect.label:<24}{mark}  (占比 {ratio:.2f})")

    print("-" * 84)
    print(f"{ok}/{len(SAMPLES)} 通过，样张已写入 {out_dir.resolve()}")

    if ok != len(SAMPLES):
        print("\n提示: 未通过的用例请检查 _COLOR_RANGES 与该底色的 HSV 区间是否匹配")

    return 0 if ok == len(SAMPLES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
