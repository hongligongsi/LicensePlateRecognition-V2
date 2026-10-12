"""在**无标注的真实图片**上做弱监督定位评测。

为什么需要它
------------
CCPD 镜像的标注在重命名时被抹掉了（`sample_N.jpg`），公开带标注集又下不完整，
所以我们手上大量**真实**图片没有真值框。此时：
* 用合成图算准确率 → 自欺欺人（实测合成牌只有超大才被定位到，见 UPGRADE.md）
* 直接不评 → 改定位参数无人拦（上一轮"调参"就是这么在 6 张合成图上得出
  `morph_kernel_ratio=(0.025, 0.60)` 这种退化解的）

本办法用**弱监督**：真实车牌号码必须满足国标语法（省份简称 + 发牌机关字母 +
序号），所以"**能否产出语法合法的号码**"就是一个不需要人工标注的有效信号。
它测的不是 IoU，而是"定位有没有把 OCR 送对地方"—— 对调参而言这正是我们关心的。

诚实说明局限：它只能发现"明显变差"（合法率从 0.6 掉到 0.2），
不能替代带标注数据集上的精确评测，也不能反映"框贴合度"。

用法::

    python tools/eval_weak.py --images datasets/ccpd_raw --limit 60
    python tools/eval_weak.py --images datasets/ccpd_raw --limit 60 \\
        --override '{"morph_kernel_ratio": [0.005, 0.021]}'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lpr.config import Config, LocateConfig  # noqa: E402
from lpr.plate_rules import is_valid  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _imread(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def collect(root: Path, limit: int) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    return files[:limit] if limit else files


def select(files: list[Path], skip: int, limit: int) -> list[Path]:
    """按 ``skip`` / ``limit`` 切出评测子集。

    口径：**先取全量、后切片**。``--skip 60 --limit 100`` 必须是"第 61~160 张"，
    而不是"全量的前 100 张里再跳过 60"——后者实际只评 40 张，且调参段
    （前 60）与验证段会静默重叠。此 bug 曾真实发生过，故有专测看守。
    """
    if skip <= 0 and limit <= 0:
        return list(files)
    start = max(0, skip)
    end = start + limit if limit > 0 else len(files)
    return list(files[start:end])


def evaluate(files: list[Path], cfg: LocateConfig, engine: str) -> dict:
    """返回弱监督指标。"""
    from lpr.engines import get_engine

    eng = get_engine(engine, _engine_cfg(cfg))

    n = 0
    n_cand = 0            # 至少有一个候选
    n_valid = 0           # 产出了语法合法的号码
    n_multi = 0           # 产出多个候选（可能过检）
    valid_texts: list[str] = []
    t0 = time.time()

    for path in files:
        img = _imread(path)
        if img is None:
            continue
        n += 1
        try:
            results = eng.predict(img)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {path.name}: {exc}", file=sys.stderr)
            continue

        if results:
            n_cand += 1
        if len(results) > 1:
            n_multi += 1
        good = [r.text for r in results if r.text and is_valid(r.text)]
        if good:
            n_valid += 1
            valid_texts.append(good[0])

    dt = time.time() - t0
    return {
        "n_images": n,
        "n_with_candidate": n_cand,
        "n_valid_text": n_valid,
        "valid_rate": round(n_valid / n, 4) if n else 0.0,
        "candidate_rate": round(n_cand / n, 4) if n else 0.0,
        "over_detect_rate": round(n_multi / n, 4) if n else 0.0,
        "seconds": round(dt, 1),
        "ms_per_image": round(dt / n * 1000, 1) if n else 0.0,
        "sample_texts": valid_texts[:8],
    }


def _engine_cfg(locate: LocateConfig):
    from lpr.config import EngineConfig

    return EngineConfig(locate=locate)


def main() -> int:
    ap = argparse.ArgumentParser(description="无标注真实图上的弱监督定位评测")
    ap.add_argument("--images", default="datasets/ccpd_raw", help="图片目录")
    ap.add_argument("--limit", type=int, default=60, help="取前 N 张（0=全部）")
    ap.add_argument("--skip", type=int, default=0,
                    help="跳过前 N 张（留出集验证：调参用前 60，验证用 --skip 60）")
    ap.add_argument("--engine", default="deep", choices=["cv", "deep"],
                    help="用哪个引擎；cv 快但合法率本身很低，建议 deep")
    ap.add_argument("--override", default=None,
                    help='JSON 覆盖 locate 参数，如 \'{"min_area_ratio": 0.0008}\'')
    ap.add_argument("--json", metavar="PATH", help="结果写文件")
    args = ap.parse_args()

    root = Path(args.images)
    if not root.is_absolute():
        root = ROOT / root
    if not root.is_dir():
        print(f"目录不存在: {root}", file=sys.stderr)
        return 1
    files = select(collect(root, 0), args.skip, args.limit)
    if not files:
        print(f"目录下没有图片: {root}", file=sys.stderr)
        return 1

    base = Config.load().engine.locate
    cand = base
    if args.override:
        ov = json.loads(args.override)
        for k, v in ov.items():
            if isinstance(v, list):
                v = tuple(v)
            ov[k] = v
        cand = replace(base, **ov)
        print(f"覆盖参数: {args.override}")

    print(f"引擎={args.engine}  图片={len(files)}")
    print("\n[当前配置] ", end="", flush=True)
    b = evaluate(files, base, args.engine)
    print(json.dumps(b, ensure_ascii=False))

    out = {"engine": args.engine, "n_images": len(files), "current": b}
    if args.override:
        print("[候选配置] ", end="", flush=True)
        c = evaluate(files, cand, args.engine)
        print(json.dumps(c, ensure_ascii=False))
        d_rate = c["valid_rate"] - b["valid_rate"]
        print(
            f"\n{'=' * 70}\n"
            f"合法率 {b['valid_rate']:.4f} -> {c['valid_rate']:.4f}  ({d_rate:+.4f})\n"
            f"候选率 {b['candidate_rate']:.4f} -> {c['candidate_rate']:.4f}\n"
            f"过检率 {b['over_detect_rate']:.4f} -> {c['over_detect_rate']:.4f}"
        )
        if d_rate < 0:
            print("\n✗ 弱监督指标变差，不建议采用该参数")
        elif d_rate == 0:
            print("\n= 弱监督指标持平（可能是本集区分度不够，或改动确实中性）")
        else:
            print("\n✓ 弱监督指标变好（仍需带标注数据集确认）")
        out["candidate"] = c
        out["delta_valid_rate"] = round(d_rate, 4)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
