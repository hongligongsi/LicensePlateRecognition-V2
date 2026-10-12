"""稳健下载 CCPD 样本图片（ModelScope 镜像，编号连续）。

背景
----
早期版本用 ``dataset_file_download`` 逐张下载并共用一个 cache 目录，
进程会在大约 1 分钟后被静默终止（不是异常，``except Exception`` 抓不到），
导致每轮只能下 25~80 张。

本版本做了三件事：

1. **每批独立 cache 目录** —— 批内下载完立即整体删除，避免任何跨批累积状态。
2. **捕获 BaseException** —— 连 ``SystemExit`` / ``KeyboardInterrupt`` 一起记录，
   让"被杀"这件事留下证据而不是无声消失。
3. **逐批写日志并 flush** —— 用 ``--start`` 续跑时能明确知道断点。

用法::

    python tools/fetch_ccpd.py --total 1000 --start 360 --batch 20
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUT = DATA / "ccpd_raw"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_batch(indices: list[int], cache: Path) -> tuple[int, int, list[str]]:
    """下载一批到 *cache*（批内复用，结束后由调用方删除）。"""
    from modelscope import dataset_file_download

    ok = 0
    errs: list[str] = []
    for i in indices:
        dst = OUT / f"sample_{i}.jpg"
        if dst.exists() and dst.stat().st_size > 500:
            continue
        try:
            p = dataset_file_download(
                "OmniData/CCPD",
                f"sample/image/sample_{i}.jpg",
                cache_dir=str(cache),
            )
            if Path(p).exists() and Path(p).stat().st_size > 500:
                shutil.move(str(p), str(dst))
                ok += 1
            else:
                errs.append(f"{i}:空文件")
        except BaseException as exc:  # noqa: BLE001 - 故意抓 BaseException
            errs.append(f"{i}:{type(exc).__name__}")
    return ok, len(indices), errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1000, help="目标总张数")
    ap.add_argument("--start", type=int, default=0, help="起始编号")
    ap.add_argument("--batch", type=int, default=20, help="每批张数")
    ap.add_argument("--max-batch", type=int, default=200, help="最多跑多少批（安全阀）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_ok = 0
    i = args.start
    batches = 0

    while i < args.total and batches < args.max_batch:
        end = min(i + args.batch, args.total)
        idx = [j for j in range(i, end)
               if not (OUT / f"sample_{j}.jpg").exists()]
        if not idx:
            i = end
            continue

        cache = DATA / f"_cache_{batches}"
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(parents=True, exist_ok=True)

        try:
            ok, n, errs = download_batch(idx, cache)
            total_ok += ok
            have = len(list(OUT.glob("sample_*.jpg")))
            log(f"批次 {i}-{end - 1}  本批 {ok}/{n}  累计落盘 {have}  "
                f"用时 {time.time() - t0:.0f}s")
            if errs:
                log(f"    失败 {len(errs)}: {errs[:4]}")
        except BaseException as exc:  # noqa: BLE001
            log(f"批次 {i}-{end - 1} 整体中断: {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(cache, ignore_errors=True)

        i = end
        batches += 1

    have = len(list(OUT.glob("sample_*.jpg")))
    log(f"结束: 本轮新下 {total_ok} 张, 目录共 {have} 张, "
        f"耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
