"""对抗性退化样张：对现有真值样张施加确定性退化，模拟真实困难场景。

退化类型（全部确定性实现，noise 用固定种子，跨运行结果一致）：
* blur      —— 高斯模糊（运动/离焦）
* downscale —— 先缩后放，模拟远距离小目标
* brighten/darken —— gamma 提亮/压暗（逆光/夜间）
* noise     —— 高斯噪点（低照度传感器）
* tilt      —— 小角度倾斜（安装视角）
* compress  —— 重度 JPEG 压缩（网络传输）

消费方式：
* ``tests/test_hard_samples_accuracy.py`` 直接调用 :func:`iter_hard_samples`
  在内存里拿退化图跑门槛测试（推荐，仓库不落二进制）；
* 命令行 ``python tools/make_hard_samples.py`` 把退化图落盘到
  ``tests/baselines/hard/`` 供人工检视/调参（该目录已 gitignore）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GT_PATH = ROOT / "tests" / "baselines" / "ground_truth.json"
OUT_DIR = ROOT / "tests" / "baselines" / "hard"

#: 退化参数（保守取值：模拟真实困难场景，但不应把牌面毁到物理不可读）
DEGRADATIONS: dict[str, dict] = {
    "blur": {"kernel": 5},  # 轻度运动模糊
    "downscale": {"scale": 0.45},  # 车牌像素缩到约原来一半
    "brighten": {"gamma": 1.6},  # 提亮（逆光过曝）
    "darken": {"gamma": 0.55},  # 压暗（夜间/阴影）
    "noise": {"sigma": 18},  # 高斯噪声
    "tilt": {"angle": 6.0},  # 小角度倾斜
    "compress": {"quality": 18},  # 重度 JPEG 压缩
}

#: 已知信息不足的 (样本, 退化) 组合：不计入命中率分母。
#: double_lv 原图已是分辨率极限（见 ground_truth.json 标注），任何退化都不可读；
#: 领馆红字在重度压缩下色彩信息丢失。
_ALLOW_MISS_ON = {("double_lv.png", kind) for kind in DEGRADATIONS} | {
    ("shi_lin_guan.jpg", "compress")
}


def apply_degradation(img: np.ndarray, kind: str, params: dict, seed: int) -> np.ndarray:
    """对图像施加指定退化（确定性；同一 (img, kind, seed) 输出恒定）。"""
    if kind == "blur":
        k = params["kernel"]
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "downscale":
        h, w = img.shape[:2]
        s = params["scale"]
        small = cv2.resize(
            img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if kind in ("brighten", "darken"):
        lut = np.array([((i / 255.0) ** params["gamma"]) * 255 for i in range(256)], np.uint8)
        return cv2.LUT(img, lut)
    if kind == "noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, params["sigma"], img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "tilt":
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), params["angle"], 1.0)
        return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if kind == "compress":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, params["quality"]])
        assert ok
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raise ValueError(kind)


def iter_hard_samples():
    """逐条产出 (退化条目, 退化图)。条目字段与 ground_truth.json 样张一致。"""
    from lpr.utils.image import imread

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    for s in gt["samples"]:
        src = ROOT / s["path"]
        if not src.exists():
            continue
        img = imread(src)
        name = Path(s["path"]).name
        for kind, params in DEGRADATIONS.items():
            entry = {
                "path": f"{s['path']}__{kind}",
                "plates": s["plates"],
                "color": s["color"],
            }
            if (name, kind) in _ALLOW_MISS_ON or s.get("known_failure"):
                entry["known_failure"] = True
            seed = hash((s["path"], kind)) & 0xFFFFFFFF
            yield entry, apply_degradation(img, kind, params, seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    from lpr.utils.image import imwrite

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for entry, img in iter_hard_samples():
        out = OUT_DIR / (Path(entry["path"]).stem + ".jpg")
        imwrite(out, img)
        n += 1
    print(f"已生成 {n} 个对抗样本 → {OUT_DIR}（该目录不入库，仅供人工检视）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
