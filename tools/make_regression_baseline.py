"""生成"真实样张行为快照"基线，供 CI 做回归拦截。

为什么是"快照"而不是"准确率"
------------------------------
真实带标注的数据集（CCPD 的标注在镜像里被抹掉；happygiraffe 那套只有 320 张）
都不适合进仓库：体积大、再分发有版权顾虑。

而**合成图不能当真值用** —— 我们实测过：合成牌只有放大到占画面一半宽度
才勉强被定位到，且找到的是字符区域而不是车牌区域。拿它算准确率是自欺欺人。

所以这里的基线是**真实样张上的行为快照**：用**确定性**的 cv 引擎
（SVM，纯 CPU，不依赖任何需下载的 OCR 模型）跑仓库里随源码分发的
``img/*.jpg`` 与 ``samples/*.jpg``，把"定位框 + 底色 + 识别文本"记下来。

它能回答的问题：**"这次改动有没有悄悄改变既有样张的行为？"**
它不能回答的问题："准确率是多少？" —— 那需要带标注的真实数据集，
请用 :mod:`tools.eval_locate` 在本地评。

用法::

    # 重新生成基线（**只在确认行为变化是预期之内时才做**）
    python tools/make_regression_baseline.py --write

    # 只看看当前行为（不改基线）
    python tools/make_regression_baseline.py
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

from lpr.config import Config
from lpr.engines.locate import locate_plates

#: 基线文件位置（进仓库）
BASELINE_PATH = ROOT / "tests" / "baselines" / "real_samples.json"

#: 参与快照的样张（相对项目根）；全部是仓库内已有、随源码分发的真实照片
SAMPLE_DIRS = ("img", "samples")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect() -> list[Path]:
    files: list[Path] = []
    for d in SAMPLE_DIRS:
        root = ROOT / d
        if not root.is_dir():
            continue
        files.extend(sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXT))
    return files


def _imread(path: Path) -> np.ndarray | None:
    """读图（cv2.imread 不支持中文路径，用 fromfile 兜底）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def _rel(path: Path) -> str:
    """相对项目根的路径；不在项目内（例如临时目录）则退回绝对路径。"""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def snapshot() -> dict:
    """跑一遍确定性定位，产出行为快照。"""
    cfg = Config.load()
    locate_cfg = cfg.engine.locate
    records = []

    for path in collect():
        img = _imread(path)
        if img is None:
            records.append({"path": _rel(path), "readable": False})
            continue

        cands, _ = locate_plates(img, locate_cfg, None)
        top = cands[0] if cands else None
        records.append(
            {
                "path": _rel(path),
                "readable": True,
                "size": [int(img.shape[1]), int(img.shape[0])],
                "n_candidates": len(cands),
                # 本项目 bbox 口径是 (x, y, w, h)
                "top_bbox": [int(v) for v in top.bbox] if top is not None else None,
                "top_color": top.color.value if top is not None else None,
            }
        )
    return {
        "_note": (
            "真实样张行为快照，用于 CI 回归拦截；不是准确率指标。"
            "确认行为变化属预期后，用 tools/make_regression_baseline.py --write 更新。"
        ),
        "_engine": "cv (deterministic)",
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="生成/查看真实样张行为快照基线")
    ap.add_argument("--write", action="store_true", help="把快照写入基线文件")
    ap.add_argument("--out", default=str(BASELINE_PATH), help="基线文件路径")
    args = ap.parse_args()

    snap = snapshot()
    text = json.dumps(snap, ensure_ascii=False, indent=2)

    if args.write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"基线已写入 {out}（{len(snap['records'])} 条）")
        return 0

    readable = [r for r in snap["records"] if r.get("readable")]
    with_box = [r for r in readable if r.get("top_bbox")]
    print(
        f"样张 {len(snap['records'])} 张（可读 {len(readable)}），其中定位到候选 {len(with_box)} 张"
    )
    for r in snap["records"]:
        if not r.get("readable"):
            print(f"  ! 无法读取 {r['path']}")
            continue
        bbox = r["top_bbox"]
        print(
            f"  {r['path']:<34} {r['size'][0]}x{r['size'][1]:<6} "
            f"候选={r['n_candidates']:<3} "
            f"top={bbox if bbox else '—'}  色={r['top_color'] or '—'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
