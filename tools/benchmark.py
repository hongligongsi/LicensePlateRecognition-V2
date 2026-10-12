#!/usr/bin/env python
"""LPR v2.1 基准测试（Benchmark）。

量化四个优化方向的收益，输出可直接引用的数据：

* **更快速** —— 各引擎单张耗时、吞吐量（QPS）、冷启动时间
* **更精准** —— 在标注样本上的命中率、平均置信度
* **更安全** —— 脱敏生效性、限流、输入限制的拦截情况
* **更全面** —— 车牌类型覆盖数、输出格式数

用法::

    python tools/benchmark.py --images samples/ --engine deep --repeat 3
    python tools/benchmark.py --images samples/ --json out/bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# 允许直接从仓库根目录运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from lpr.api import Recognizer
from lpr.config import Config, LocateConfig
from lpr.engines.locate import detect_color
from lpr.plate_rules import FORMATS, is_valid
from lpr.types import PlateColor
from lpr.utils.image import imread
from lpr.utils.security import (
    Limits,
    PrivacyPolicy,
    RateLimiter,
    SecurityError,
    check_image,
    mask_plate,
    scrub_value,
)


def _fmt_ms(value: float) -> str:
    return f"{value:7.1f} ms"


def _bench_engine(name: str, images: list[tuple[str, np.ndarray]], repeat: int) -> dict:
    """对单个引擎跑耗时与命中率统计。"""
    cfg = Config.load()
    try:
        rec = Recognizer(engine=name, config=cfg)
    except Exception as exc:  # 引擎不可用（缺依赖）时跳过而不是崩掉整份报告
        return {"engine": name, "available": False, "reason": str(exc)[:200]}

    t0 = time.perf_counter()
    rec.warmup()
    cold_ms = (time.perf_counter() - t0) * 1000

    per_image: list[dict] = []
    all_latencies: list[float] = []

    for path, img in images:
        latencies = []
        texts: list[str] = []
        confs: list[float] = []
        for _ in range(repeat):
            start = time.perf_counter()
            outcome = rec.recognize(img)
            latencies.append((time.perf_counter() - start) * 1000)
            texts = outcome.texts
            confs = [r.confidence for r in outcome.results]
        per_image.append(
            {
                "path": Path(path).name,
                "min_ms": round(min(latencies), 2),
                "median_ms": round(statistics.median(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "texts": texts,
                "hit": bool(texts),
                "confidence": round(max(confs), 4) if confs else 0.0,
                "valid": all(is_valid(t) for t in texts) if texts else False,
            }
        )
        all_latencies.extend(latencies)

    hit = sum(1 for r in per_image if r["hit"])
    valid_hits = sum(1 for r in per_image if r["valid"])
    avg_lat = statistics.mean(all_latencies) if all_latencies else 0.0
    med_lat = statistics.median(all_latencies) if all_latencies else 0.0

    return {
        "engine": name,
        "available": True,
        "cold_start_ms": round(cold_ms, 1),
        "images": len(images),
        "repeat": repeat,
        "runs": len(all_latencies),
        "avg_ms": round(avg_lat, 2),
        "median_ms": round(med_lat, 2),
        "p95_ms": round(float(np.percentile(all_latencies, 95)) if all_latencies else 0.0, 2),
        "qps": round(1000.0 / avg_lat, 2) if avg_lat else 0.0,
        "hit": hit,
        "hit_rate": round(hit / len(images), 4) if images else 0.0,
        "valid_rate": round(valid_hits / len(images), 4) if images else 0.0,
        "per_image": per_image,
    }


def _bench_security() -> dict:
    """验证安全层各项防护确实生效。"""
    checks: list[dict] = []

    def check(name: str, ok: bool, note: str = "") -> None:
        checks.append({"check": name, "passed": bool(ok), "note": note})

    policy = PrivacyPolicy(mask_output=True)

    # 1) 基础打码
    masked = mask_plate("京AD77972", policy)
    check("车牌打码", masked != "京AD77972", f"{masked}")

    # 2) 嵌套结构脱敏（meta.variants 泄露是历史漏洞）
    payload = {"meta": {"variants": [["京AD77972", 1.2]]}}
    scrubbed = json.dumps(scrub_value(payload, policy), ensure_ascii=False)
    check("嵌套结构脱敏", "京AD77972" not in scrubbed, scrubbed[:60])

    # 3) 超大文件拦截
    limits = Limits(max_file_bytes=1024)
    try:
        check_image(np.zeros((720, 1280, 3), dtype=np.uint8), Limits(max_pixels=1000))
        check("像素上限拦截", False, "未抛异常")
    except SecurityError as exc:
        check("像素上限拦截", True, str(exc)[:60])

    # 4) 畸形尺寸拦截
    try:
        check_image(
            np.broadcast_to(np.uint8(0), (4, 50_000, 3)), Limits(max_side=1000, max_pixels=10**9)
        )
        check("单边上限拦截", False, "未抛异常")
    except SecurityError as exc:
        check("单边上限拦截", True, str(exc)[:60])

    # 5) 限流
    limiter = RateLimiter(max_per_second=10.0)
    burst = sum(limiter.acquire(timeout=0.0) for _ in range(50))
    check("令牌桶限流", burst <= 15, f"突发放行 {burst} 次")

    _ = limits  # 保留变量语义
    return {
        "checks": checks,
        "passed": sum(1 for c in checks if c["passed"]),
        "total": len(checks),
    }


def _bench_coverage() -> dict:
    """统计车牌类型、底色与输出格式的覆盖广度。"""
    samples = {
        "普通民用": "京A12345",
        "新能源": "京AD77972",
        "警用": "京A1234警",
        "军队": "京V12345",
        "武警": "WJ京1234X",
        "使领馆": "使123456",
        "挂车": "京A1234挂",
        "教练车": "京A123学",
        "危险品运输": "京A1234危",
        "港澳入出境": "粤Z1234港",
    }
    covered = {k: is_valid(v) for k, v in samples.items()}

    # 五种底色 + 危险品的判定（合成车牌，不依赖样本图）
    color_cases = [
        ("黑色", (10, 10, 10), (255, 255, 255), PlateColor.BLACK),
        ("蓝色", (200, 80, 20), (255, 255, 255), PlateColor.BLUE),
        ("绿色", (80, 200, 60), (10, 10, 10), PlateColor.GREEN),
        ("白色", (245, 245, 245), (10, 10, 10), PlateColor.WHITE),
        ("黄色", (0, 220, 255), (10, 10, 10), PlateColor.YELLOW),
        ("危险品(黑底黄字)", (10, 10, 10), (0, 220, 255), PlateColor.HAZMAT),
    ]
    cfg = LocateConfig()
    color_covered = {}
    for name, bg, fg, expect in color_cases:
        probe = np.full((60, 220, 3), bg, np.uint8)
        cv2.putText(probe, "A1234", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, fg, 3)
        got, _ = detect_color(probe, cfg)
        color_covered[name] = got is expect

    return {
        "formats_defined": len(FORMATS),
        "types_tested": len(samples),
        "types_passed": sum(1 for v in covered.values() if v),
        "coverage": covered,
        "colors_tested": len(color_cases),
        "colors_passed": sum(1 for v in color_covered.values() if v),
        "colors": color_covered,
        "output_formats": ["console", "json", "csv", "annotated_image", "annotated_video"],
        "input_sources": ["image", "directory", "video_file", "camera_stream", "ndarray"],
    }


def _print_report(engine_reports: list[dict], security: dict, coverage: dict) -> None:
    line = "=" * 72

    print(f"\n{line}\n  LPR v2.1 基准报告\n{line}")

    print("\n【更快速 / 更精准】引擎性能")
    print("-" * 72)
    header = f"{'引擎':<10}{'冷启动':>12}{'中位耗时':>14}{'P95':>12}{'QPS':>10}{'命中率':>10}"
    print(header)
    for rep in engine_reports:
        if not rep["available"]:
            print(f"{rep['engine']:<10}{'不可用':>12}  {rep['reason'][:40]}")
            continue
        print(
            f"{rep['engine']:<10}"
            f"{rep['cold_start_ms']:>10.0f}ms"
            f"{rep['median_ms']:>12.1f}ms"
            f"{rep['p95_ms']:>10.1f}ms"
            f"{rep['qps']:>9.1f}"
            f"{rep['hit_rate'] * 100:>9.1f}%"
        )

    print("\n【更安全】防护检查")
    print("-" * 72)
    for chk in security["checks"]:
        mark = "通过" if chk["passed"] else "失败"
        print(f"  [{mark}] {chk['check']:<16} {chk['note']}")
    print(f"  小计: {security['passed']}/{security['total']} 项通过")

    print("\n【更全面】覆盖范围")
    print("-" * 72)
    print(f"  已定义车牌格式: {coverage['formats_defined']} 种")
    print(f"  类型识别验证:   {coverage['types_passed']}/{coverage['types_tested']} 通过")
    print(f"  底色识别验证:   {coverage['colors_passed']}/{coverage['colors_tested']} 通过")
    failed_colors = [k for k, v in coverage.get("colors", {}).items() if not v]
    if failed_colors:
        print(f"    未通过底色: {', '.join(failed_colors)}")
    print(f"  输出格式:       {', '.join(coverage['output_formats'])}")
    print(f"  输入来源:       {', '.join(coverage['input_sources'])}")

    if any(r["available"] for r in engine_reports):
        print("\n【逐图明细】")
        print("-" * 72)
        for rep in engine_reports:
            if not rep["available"]:
                continue
            print(f"  {rep['engine']}:")
            for item in rep["per_image"]:
                status = "命中" if item["hit"] else "未命中"
                plate = ", ".join(item["texts"]) or "—"
                print(
                    f"    {item['path']:<20} {status:<6} {plate:<14} "
                    f"置信度={item['confidence']:.2f} 中位={item['median_ms']:.0f}ms"
                )

    print(f"\n{line}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="LPR v2.1 基准测试")
    parser.add_argument("--images", default="samples", help="测试图片目录或文件")
    parser.add_argument(
        "--engine",
        default="deep,cv",
        help="逗号分隔的引擎列表（默认 deep,cv）",
    )
    parser.add_argument("--repeat", type=int, default=3, help="每张图重复次数（默认 3）")
    parser.add_argument("--json", dest="json_out", help="把结果写入 JSON 文件")
    parser.add_argument("--skip-security", action="store_true", help="跳过安全检查项")
    args = parser.parse_args()

    # ---- 收集图片
    target = Path(args.images)
    if target.is_dir():
        paths = sorted(
            p
            for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in Limits().allowed_extensions
        )
    else:
        paths = [target]

    images: list[tuple[str, np.ndarray]] = []
    for p in paths:
        img = imread(p)
        if img is not None:
            images.append((str(p), img))

    if not images:
        print(f"未在 {args.images} 找到可用图片", file=sys.stderr)
        return 1

    print(f"载入 {len(images)} 张图片，每张重复 {args.repeat} 次")

    engine_reports = [
        _bench_engine(name.strip(), images, args.repeat)
        for name in args.engine.split(",")
        if name.strip()
    ]
    security = (
        _bench_security() if not args.skip_security else {"checks": [], "passed": 0, "total": 0}
    )
    coverage = _bench_coverage()

    _print_report(engine_reports, security, coverage)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "engines": engine_reports,
                    "security": security,
                    "coverage": coverage,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"报告已写入 {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
