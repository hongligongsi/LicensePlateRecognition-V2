"""命令行入口。

用法示例::

    # 识别单张图片
    lpr predict samples/1.jpg

    # 识别整个目录
    lpr predict samples/ --engine deep

    # 视频文件（抽帧识别）
    lpr video parking.mp4

    # 启动图形界面
    lpr gui

    # 查看可用引擎
    lpr engines
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Config
from .engines import describe_engines, get_engine
from .types import PlateResult
from .utils.image import draw_result, imwrite
from .utils.logging_setup import setup_logging


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lpr",
        description="中国车牌识别 v2 —— 传统 CV + 深度学习双引擎",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 2.0.0")

    sub = parser.add_subparsers(dest="command", required=True)

    # predict
    p_pred = sub.add_parser("predict", help="识别图片（单张/目录）")
    p_pred.add_argument("inputs", nargs="+", help="图片路径或目录")
    p_pred.add_argument("--engine", default=None, choices=["cv", "deep"], help="识别引擎")
    p_pred.add_argument("--save-annotated", action="store_true", help="保存标注图")
    p_pred.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    p_pred.add_argument("--recursive", action="store_true", help="目录时递归查找")

    # video
    p_video = sub.add_parser("video", help="识别视频文件")
    p_video.add_argument("input", help="视频路径")
    p_video.add_argument("--engine", default=None, choices=["cv", "deep"])
    p_video.add_argument("--stride", type=int, default=15, help="每 N 帧识别一次")
    p_video.add_argument("--output", default=None, help="输出标注视频路径")

    # gui
    sub.add_parser("gui", help="启动图形界面")

    # engines
    sub.add_parser("engines", help="查看可用引擎")

    # config
    p_cfg = sub.add_parser("config", help="生成默认配置文件")
    p_cfg.add_argument("--path", default=None, help="写入位置（默认 config.json）")

    return parser


def _collect_images(inputs: list[str], recursive: bool) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            out.extend(p for p in path.glob(pattern) if p.suffix.lower() in exts)
        else:
            out.append(path)
    return sorted(out)


def cmd_predict(args: argparse.Namespace, cfg: Config) -> int:
    setup_logging(cfg.log_level)
    engine_name = args.engine or cfg.engine.default
    engine = get_engine(engine_name, cfg.engine)

    images = _collect_images(args.inputs, args.recursive)
    if not images:
        print("未找到图片文件", file=sys.stderr)
        return 1

    total_ms = 0.0
    hits = 0
    payloads = []

    for path in images:
        started = time.perf_counter()
        try:
            results: list[PlateResult] = engine.predict(path)
        except Exception as exc:
            print(f"[错误] {path}: {exc}", file=sys.stderr)
            continue
        elapsed = (time.perf_counter() - started) * 1000
        total_ms += elapsed

        if results:
            hits += 1
            for res in results:
                mark = "✓" if res.confidence >= 0.7 else "?"
                print(f"{mark} {path.name}: {res}")

            if args.save_annotated:
                img = draw_result(_load(path), results)
                out = Path("annotated") / path.name
                imwrite(out, img)
                print(f"    标注图 → {out}")
        else:
            print(f"- {path.name}: 未识别到车牌 ({elapsed:.0f} ms)")

        if args.json:
            payloads.append({"image": str(path), "results": [r.to_dict() for r in results]})

    print(f"\n共 {len(images)} 张，识别出车牌 {hits} 张，平均 {total_ms / len(images):.0f} ms/张")

    if args.json:
        print(json.dumps(payloads, ensure_ascii=False, indent=2))
    return 0


def _load(path: Path):
    from .utils.image import imread

    return imread(path)


def cmd_video(args: argparse.Namespace, cfg: Config) -> int:
    import cv2

    setup_logging(cfg.log_level)
    engine_name = args.engine or cfg.engine.default
    engine = get_engine(engine_name, cfg.engine)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"无法打开视频: {args.input}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    seen: set[str] = set()
    frame_idx = 0
    started_all = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame_idx % args.stride == 0:
            for res in engine.predict(frame):
                if res.text and res.text not in seen:
                    seen.add(res.text)
                    sec = frame_idx / fps
                    print(f"[{sec:8.2f}s] #{frame_idx}: {res}")
                if writer is not None:
                    frame = draw_result(frame, [res])
        if writer is not None:
            writer.write(frame)

    cap.release()
    writer and writer.release()
    elapsed = time.perf_counter() - started_all
    print(f"\n处理 {frame_idx} 帧 / {elapsed:.1f}s，识别到 {len(seen)} 个不同车牌")
    return 0


def cmd_gui(args: argparse.Namespace, cfg: Config) -> int:
    from .ui.window import run_ui

    run_ui(cfg)
    return 0


def cmd_engines(args: argparse.Namespace, cfg: Config) -> int:
    rows = describe_engines()
    print(f"{'引擎':<8} {'状态':<6} 说明")
    for row in rows:
        status = "可用" if row["available"] else "不可用"
        print(f"{row['display_name']:<8} {status:<6} {row['description']}")
        if not row["available"]:
            print(f"{'':<16} 原因: {row['reason']}")
    return 0


def cmd_config(args: argparse.Namespace, cfg: Config) -> int:
    from .config import DEFAULT_CONFIG_PATH

    path = cfg.save(args.path or DEFAULT_CONFIG_PATH)
    print(f"默认配置已写入: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    cfg = Config.load()
    handlers = {
        "predict": cmd_predict,
        "video": cmd_video,
        "gui": cmd_gui,
        "engines": cmd_engines,
        "config": cmd_config,
    }
    return handlers[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
