"""批量识别多进程加速基准。

用途：在**你自己的机器**上判断 ``--workers`` 能带来多少收益。
本仓库的 CI/沙箱环境 CPU 配额受限（可能只分到 1 个核），在那里并行会更慢；
换到真实多核机器上再跑这个脚本，才是有意义的结论。

用法::

    python tools/bench_workers.py --images datasets/ccpd_raw --limit 160 \
        --engine deep --workers 1 4 8

    # 也可以同时对照测一下"纯 CPU 计算"的多进程扩展性，
    # 用来判断瓶颈到底在 CPU 配额还是在识别本身
    python tools/bench_workers.py --cpu-control
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpr.api import Recognizer  # noqa: E402
from lpr.batch import _cpu_budget, resolve_workers  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: Path, limit: int) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return files[:limit] if limit else files


def bench(engine: str, files: list[Path], workers: int, repeat: int) -> dict:
    times: list[float] = []
    hits = plates = 0
    for _ in range(repeat):
        rec = Recognizer(engine=engine)
        rec.warmup()
        t0 = time.perf_counter()
        batch = rec.recognize_many(files, workers=workers)
        times.append(time.perf_counter() - t0)
        hits = sum(1 for o in batch.outcomes if o.results)
        plates = sum(len(o.results) for o in batch.outcomes)

    med = statistics.median(times)
    return {
        "workers": workers,
        "real_workers": resolve_workers(workers, len(files)),
        "seconds": round(med, 3),
        "ms_per_image": round(med / max(1, len(files)) * 1000, 1),
        "hits": hits,
        "plates": plates,
    }


def burn(n: int) -> int:
    """纯 CPU 计算，用来探测环境的真实并行扩展性。"""
    x = 0
    for i in range(n):
        x += i * i % 7
    return x


def cpu_control(processes: int = 4, n: int = 3_000_000) -> None:
    print(f"\n[对照] 纯 CPU 计算，可调度 CPU={_cpu_budget()}")
    t0 = time.perf_counter()
    burn(n * processes)
    serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(processes) as pool:
        pool.map(burn, [n] * processes)
    par = time.perf_counter() - t0
    ratio = serial / par if par else 0.0
    print(f"  {processes}x 工作量：串行 {serial:.2f}s / {processes} 进程 {par:.2f}s → {ratio:.2f}x")
    if ratio < 1.2:
        print("  ⚠ 环境几乎没有并行余量（CPU 配额受限），此时 workers>1 只会更慢。")
        print("    这不是实现的锅 —— 上面的纯 CPU 计算同样没有扩展性。")


def main() -> int:
    ap = argparse.ArgumentParser(description="批量识别多进程加速基准")
    ap.add_argument("--images", default="datasets/ccpd_raw", help="图片目录")
    ap.add_argument("--limit", type=int, default=160, help="最多取多少张（0=全部）")
    ap.add_argument("--engine", default="deep", choices=["cv", "deep", "cascade"])
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4], help="要对比的进程数")
    ap.add_argument("--repeat", type=int, default=1, help="每个配置重复次数（取中位）")
    ap.add_argument("--json", metavar="PATH", help="把结果写成 JSON")
    ap.add_argument("--cpu-control", action="store_true", help="附加纯 CPU 并行扩展性对照")
    args = ap.parse_args()

    if args.cpu_control:
        cpu_control()

    root = Path(args.images)
    if not root.is_dir():
        print(f"图片目录不存在: {root}", file=sys.stderr)
        return 1
    files = collect_images(root, args.limit)
    if not files:
        print(f"目录下没有图片: {root}", file=sys.stderr)
        return 1

    print(
        f"\n引擎={args.engine}  图片={len(files)}  可调度CPU={_cpu_budget()} "
        f"  os.cpu_count={os.cpu_count()}"
    )
    rows = []
    base: float | None = None
    for w in args.workers:
        row = bench(args.engine, files, w, args.repeat)
        if base is None:
            base = row["seconds"]
        row["speedup"] = round(base / row["seconds"], 2) if row["seconds"] else 0.0
        rows.append(row)
        print(
            f"  workers={row['workers']:<3} (实际 {row['real_workers']}) "
            f"{row['seconds']:7.2f}s  {row['ms_per_image']:7.1f} ms/张  "
            f"命中图 {row['hits']}/{len(files)}  号码 {row['plates']}  {row['speedup']}x"
        )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "engine": args.engine,
                    "images": len(files),
                    "cpu_budget": _cpu_budget(),
                    "os_cpu_count": os.cpu_count(),
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
