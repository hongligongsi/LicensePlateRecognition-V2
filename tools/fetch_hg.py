"""下载带标注的车牌检测评测集（ModelScope: happygiraffe/vihicle-license-plate-datasets）。

只取 ``test`` 划分：``test/images/*.jpg`` + ``test/labels/*.txt``（YOLO 格式）。
这类数据集的价值在于**有真值框**，可用于 ``tools/eval_locate.py`` 客观评测定位质量。

为什么要写这个脚本而不是直接 ``dataset_snapshot_download(allow_file_pattern="test/*")``：
该 API 的 ``allow_file_pattern`` **不做通配展开**，传 ``test/*`` 会原地卡死。
这里改为先分页列全量文件名，再按显式路径分批下载——每批用独立 cache 目录，
避免长跑进程被静默终止后整批丢失。

用法::

    python tools/fetch_hg.py --out datasets/hg --batch 25
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "happygiraffe/vihicle-license-plate-datasets"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_files(prefix: str, api) -> list[str]:
    """分页列出一个目录下的全部文件路径。"""
    names: list[str] = []
    page = 1
    while True:
        try:
            fs = api.get_dataset_files(
                repo_id=REPO,
                revision="master",
                root_path=prefix,
                recursive=False,
                page_number=page,
                page_size=500,
            )
        except Exception as exc:
            log(f"  列目录 {prefix} 第 {page} 页失败: {type(exc).__name__}")
            break
        if not fs:
            break
        names += [f["Path"] for f in fs]
        if len(fs) < 500:
            break
        page += 1
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/hg", help="输出根目录（相对项目根）")
    ap.add_argument("--batch", type=int, default=25, help="每批文件数")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只取前 N 对（0=全部）。做小规模验证时用，避免下满 300+ 张",
    )
    args = ap.parse_args()

    out = ROOT / args.out
    img_dir = out / "test" / "images"
    lab_dir = out / "test" / "labels"
    cache_root = ROOT / "datasets" / "_hg_cache"
    for d in (img_dir, lab_dir):
        d.mkdir(parents=True, exist_ok=True)

    from modelscope import dataset_file_download
    from modelscope.hub.api import HubApi

    api = HubApi()
    log("列出文件清单 ...")
    images = [
        p for p in list_files("test/images", api) if p.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    labels = [p for p in list_files("test/labels", api) if p.endswith(".txt")]
    log(f"清单: {len(images)} 张图, {len(labels)} 个标注")

    if not images:
        log("没拿到文件清单，退出")
        return 1

    pairs = []
    for ip in images:
        stem = Path(ip).stem
        lp = f"test/labels/{stem}.txt"
        pairs.append((ip, lp))

    if args.limit > 0:
        pairs = pairs[: args.limit]
        log(f"按 --limit 截取前 {len(pairs)} 对")

    t0 = time.time()
    ok_img = ok_lab = 0
    i = 0
    while i < len(pairs):
        chunk = pairs[i : i + args.batch]
        cache = cache_root / f"b{i}"
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(parents=True, exist_ok=True)

        for rel, lab in chunk:
            dst = img_dir / Path(rel).name
            if not (dst.exists() and dst.stat().st_size > 500):
                try:
                    p = dataset_file_download(REPO, rel, cache_dir=str(cache))
                    if Path(p).exists() and Path(p).stat().st_size > 500:
                        shutil.move(str(p), str(dst))
                        ok_img += 1
                except BaseException as exc:
                    log(f"  图失败 {rel}: {type(exc).__name__}")
            ldst = lab_dir / Path(lab).name
            if not ldst.exists():
                try:
                    p = dataset_file_download(REPO, lab, cache_dir=str(cache))
                    if Path(p).exists():
                        shutil.move(str(p), str(ldst))
                        ok_lab += 1
                except BaseException:
                    pass

        shutil.rmtree(cache, ignore_errors=True)
        have_i = len(list(img_dir.glob("*")))
        have_l = len(list(lab_dir.glob("*.txt")))
        log(
            f"批次 {i}-{i + len(chunk) - 1}  图 {have_i}  标注 {have_l}  "
            f"用时 {time.time() - t0:.0f}s"
        )
        i += len(chunk)

    log(
        f"完成: 新下图 {ok_img}, 新下标注 {ok_lab}, "
        f"目录图 {len(list(img_dir.glob('*')))}, 标注 {len(list(lab_dir.glob('*.txt')))}"
    )
    log(f"评测命令: python tools/eval_locate.py {args.out}/test --grid quick")
    return 0


if __name__ == "__main__":
    sys.exit(main())
