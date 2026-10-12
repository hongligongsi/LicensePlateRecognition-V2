"""命令行入口。

用法示例::

    # 识别单张图片
    lpr predict samples/1.jpg

    # 识别整个目录（含子目录），输出标注图与 JSON 报告
    lpr predict photos/ --recursive --engine deep --save-annotated --report out/

    # 视频（跨帧跟踪去重，同一辆车只报一次）
    lpr video parking.mp4 --stride 8 --engine deep

    # 摄像头
    lpr video 0 --engine deep

    # 查看引擎状态
    lpr engines

    # 校验车牌号是否符合国家规范
    lpr check 京AD77972 京E51619

    # 生成默认配置文件
    lpr config
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import Recognizer
from .config import Config
from .engines import describe_engines
from .plate_rules import correct, describe as describe_plate, is_valid
from .types import parse_color
from .utils.logging_setup import setup_logging
from .utils.security import SecurityError, mask_plate, model_info

PROG = "lpr"
VERSION = "2.3.6"


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="中国车牌识别 v2.3 —— 双引擎 + 定位引导加速 + 全类型车牌支持",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------------------------------------------------- predict
    p = sub.add_parser("predict", help="识别图片（单张/目录）")
    p.add_argument("inputs", nargs="+", help="图片路径或目录")
    p.add_argument(
        "--engine", default=None, choices=["auto", "cv", "deep", "cascade"],
        help="识别引擎：auto(自动) / cv(传统) / deep(深度学习) / cascade(级联，最省时)",
    )
    p.add_argument("--recursive", action="store_true", help="目录时递归查找子目录")
    p.add_argument("--save-annotated", metavar="DIR", nargs="?", const="annotated",
                   help="保存标注图到指定目录（默认 annotated/）")
    p.add_argument("--report", metavar="DIR", help="输出 JSON + CSV 报告到指定目录")
    p.add_argument("--json", action="store_true", help="在终端以 JSON 输出")
    p.add_argument("--mask", action="store_true", help="对输出中的车牌号打码（隐私保护）")
    p.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help=(
            "并行识别进程数（默认 1=串行）。>1 走多进程加速，适合成百上千张；"
            "0 表示按 CPU 核数自动决定。注意：多进程模式下结果不含 ROI 图像"
        ),
    )
    p.add_argument(
        "--color",
        default=None,
        help=(
            "只保留指定底色的车牌，可多次或用逗号分隔。"
            "可选：黑色/蓝色/绿色/白色/黄色/危险品（如 --color 黄色,危险品）"
        ),
    )

    # ---------------------------------------------------------- video
    p = sub.add_parser("video", help="识别视频文件或摄像头")
    p.add_argument("input", help="视频路径，或摄像头编号（如 0）")
    p.add_argument("--engine", default=None, choices=["auto", "cv", "deep", "cascade"])
    p.add_argument("--stride", type=int, default=10, help="每 N 帧识别一次（默认 10）")
    p.add_argument("--output", default=None, help="输出标注视频路径")
    p.add_argument("--mask", action="store_true", help="输出中打码")

    # ---------------------------------------------------------- gui
    sub.add_parser("gui", help="启动图形界面")

    # ---------------------------------------------------------- engines
    sub.add_parser("engines", help="查看可用引擎状态")

    # ---------------------------------------------------------- check
    p = sub.add_parser("check", help="校验车牌号是否符合国家规范")
    p.add_argument("plates", nargs="+", help="待校验的车牌号")
    p.add_argument("--fix", action="store_true", help="尝试自动纠正不合法的号码")

    # ---------------------------------------------------------- models
    p_models = sub.add_parser("models", help="查看模型文件信息与完整性校验")
    p_models.add_argument("--verify", action="store_true", help="执行 SHA256 完整性校验")

    # ---------------------------------------------------------- config
    p_cfg = sub.add_parser("config", help="生成默认配置文件")
    p_cfg.add_argument("--path", default=None, help="写入位置（默认 config.json）")
    p_cfg.add_argument("--show", action="store_true", help="只打印当前生效配置")

    return parser


# ══════════════════════════════════════════════════════════════════
# 子命令实现
# ══════════════════════════════════════════════════════════════════


def _build_recognizer(args, cfg: Config) -> Recognizer:
    """按命令行参数构造识别器。"""
    engine = args.engine or cfg.engine.default
    if getattr(args, "mask", False):
        cfg.security.mask_output = True

    sec = cfg.security.to_context()
    sec.setup()

    return Recognizer(
        engine=engine,
        config=cfg,
        limits=sec.limits,
        privacy=sec.privacy,
    )


def cmd_predict(args: argparse.Namespace, cfg: Config) -> int:
    setup_logging(cfg.log_level)
    rec = _build_recognizer(args, cfg)
    rec.warmup()

    targets = []
    for raw in args.inputs:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*" if args.recursive else "*"
            targets.extend(
                p for p in path.glob(pattern)
                if p.is_file() and p.suffix.lower() in rec.limits.allowed_extensions
            )
        else:
            targets.append(path)

    if not targets:
        print("未找到可处理的图片", file=sys.stderr)
        return 1

    targets = sorted(targets)
    workers = int(getattr(args, "workers", 1) or 0)
    if workers != 1 and len(targets) > 1:
        from .batch import resolve_workers

        print(f"并行识别：{len(targets)} 张图 / {resolve_workers(workers, len(targets))} 进程")
    batch = rec.recognize_many(
        targets, annotate_dir=args.save_annotated, workers=workers
    )

    # 按底色过滤（可多选，如 --color 黄色,危险品）
    wanted: set | None = None
    if args.color:
        try:
            wanted = {
                parse_color(name)
                for name in str(args.color).replace("，", ",").split(",")
                if name.strip()
            }
        except ValueError as exc:
            print(f"参数错误: {exc}", file=sys.stderr)
            return 2
        before = sum(len(o.results) for o in batch.outcomes)
        for outcome in batch.outcomes:
            outcome.results = [r for r in outcome.results if r.color in wanted]
        after = sum(len(o.results) for o in batch.outcomes)
        print(f"底色过滤 {args.color}: {before} → {after} 个车牌")

    # 控制台输出同样受脱敏策略约束：开启 --mask 时不得回显明文车牌号。
    def _show(text: str) -> str:
        return mask_plate(text, rec.privacy)

    for outcome in batch.outcomes:
        name = Path(outcome.source).name
        if outcome.warnings and not outcome.results:
            print(f"! {name}: {outcome.warnings[0]}")
            continue
        if not outcome.results:
            print(f"- {name}: 未识别到车牌 ({outcome.elapsed_ms:.0f} ms)")
            continue
        for res in outcome.results:
            info = describe_plate(res.text)
            mark = "✓" if res.confidence >= 0.7 else "?"
            print(
                f"{mark} {name}: {_show(res.text)} [{res.color.label}/{info['type']}] "
                f"置信度={res.confidence:.2f} ({outcome.elapsed_ms:.0f} ms)"
            )
        for warn in outcome.warnings:
            print(f"    ⚠ {warn}")

    avg = batch.elapsed_ms / max(1, batch.total)
    print(
        f"\n共 {batch.total} 张 | 命中 {batch.hit} 张 "
        f"({batch.hit / max(1, batch.total) * 100:.1f}%) | 平均 {avg:.0f} ms/张"
    )
    unique = batch.all_texts
    if unique:
        print(f"去重后车牌: {', '.join(_show(t) for t in unique)}")

    if args.report:
        out_dir = Path(args.report)
        jp = batch.save_json(out_dir / "report.json", rec.privacy)
        cp = batch.save_csv(out_dir / "report.csv", rec.privacy)
        print(f"\n报告已写入:\n  {jp}\n  {cp}")

    if args.json:
        print(json.dumps([o.to_dict(rec.privacy) for o in batch.outcomes],
                         ensure_ascii=False, indent=2))
    return 0


def cmd_video(args: argparse.Namespace, cfg: Config) -> int:
    setup_logging(cfg.log_level)
    rec = _build_recognizer(args, cfg)
    rec.warmup()

    print(f"开始处理视频源 {args.input}（每 {args.stride} 帧识别一次，Ctrl+C 中断）")
    count = 0
    try:
        for plate in rec.recognize_video(
            args.input, stride=args.stride, output_video=args.output
        ):
            count += 1
            info = describe_plate(plate.text)
            print(
                f"[{count:3d}] {mask_plate(plate.text, rec.privacy)} "
                f"[{plate.color.label}/{info['type']}] "
                f"置信度={plate.confidence:.2f} "
                f"出现{plate.meta.get('hits')}次 一致率{plate.meta.get('hit_rate')}"
            )
    except KeyboardInterrupt:
        print("\n已中断")
    except Exception as exc:
        print(f"处理失败: {exc}", file=sys.stderr)
        return 1

    print(f"\n共确认 {count} 个不同车牌")
    if args.output:
        print(f"标注视频: {args.output}")
    return 0


def cmd_gui(args: argparse.Namespace, cfg: Config) -> int:
    from .ui.window import run_ui

    run_ui(cfg)
    return 0


def cmd_engines(args: argparse.Namespace, cfg: Config) -> int:
    print(f"{'引擎':<16} {'状态':<8} 说明")
    print("-" * 78)
    for row in describe_engines():
        status = "可用" if row["available"] else "不可用"
        print(f"{row['display_name']:<16} {status:<8} {row['description']}")
        if not row["available"]:
            print(f"{'':<16} {'':<8} 原因: {row['reason']}")
    return 0


def cmd_check(args: argparse.Namespace, cfg: Config) -> int:
    ok_count = 0
    for raw in args.plates:
        text = raw.strip().upper()
        valid = is_valid(text)
        info = describe_plate(text)
        status = "✓ 合法" if valid else "✗ 不合法"
        print(f"{text:<14} {status:<10} 类型={info['type']}")
        print(f"{'':<14} 省份={info['province'] or '—'} 发牌机关={info['issuer'] or '—'} "
              f"新能源={'是' if info['is_new_energy'] else '否'}")

        if not valid and args.fix:
            fixed, changes = correct(text)
            if fixed != text and is_valid(fixed):
                print(f"{'':<14} → 纠正为 {fixed}")
                for ch in changes:
                    print(f"{'':<16} · {ch}")
            else:
                print(f"{'':<14} → 无法自动纠正（{'; '.join(changes) or '未发现可修正项'}）")
        if valid:
            ok_count += 1

    total = len(args.plates)
    print(f"\n{ok_count}/{total} 个号码合法")
    return 0 if ok_count == total else 2


def cmd_models(args: argparse.Namespace, cfg: Config) -> int:
    from .config import MODELS_DIR

    sec = cfg.security.to_context()
    files = sorted(MODELS_DIR.glob("*")) if MODELS_DIR.exists() else []

    if not files:
        print(f"模型目录为空或不存在: {MODELS_DIR}", file=sys.stderr)
        return 1

    print(f"模型目录: {MODELS_DIR}\n")
    failed = 0
    for path in files:
        if path.is_dir():
            continue
        info = model_info(path)
        print(f"  {info['path']:<22} {info['size_bytes'] / 1048576:>8.2f} MB  "
              f"{info['mtime']}")
        if args.verify:
            check = sec.verify_models([path])[0]
            # 「跳过校验」用 —，不要显示成 ✓：没检查 ≠ 检查通过
            print(f"    {check.status.symbol} {check.message}")
            if not check.ok:
                failed += 1
    if args.verify:
        print()
        if failed:
            print(f"✗ {failed} 个模型未通过完整性校验", file=sys.stderr)
            return 1
        print("（✓=已校验通过，—=未配置哈希白名单故未校验）")
    return 0


def cmd_config(args: argparse.Namespace, cfg: Config) -> int:
    from .config import DEFAULT_CONFIG_PATH

    if args.show:
        print(json.dumps(json.loads(Path(args.path or DEFAULT_CONFIG_PATH).read_text("utf-8")),
                         ensure_ascii=False, indent=2)
              if (Path(args.path or DEFAULT_CONFIG_PATH)).exists()
              else json.dumps(cfg.save(Path(args.path) if args.path else None) and
                              json.loads((Path(args.path or DEFAULT_CONFIG_PATH)).read_text("utf-8")),
                              ensure_ascii=False, indent=2))
        return 0

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
        "check": cmd_check,
        "models": cmd_models,
        "config": cmd_config,
    }

    try:
        return handlers[args.command](args, cfg)
    except SecurityError as exc:
        print(f"安全策略拒绝: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
