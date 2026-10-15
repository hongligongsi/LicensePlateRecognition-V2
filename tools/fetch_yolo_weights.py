"""获取 / 校验 ``yolo`` 引擎的 ONNX 权重，并**检查它到底能不能被本引擎解析**。

## 为什么需要这个脚本

``yolo`` 引擎的权重不进仓库（体积 + 许可，见 ``models/yolo/README.md``）。
于是用户拿到一个 ``.onnx`` 之后会撞上三个问题：

1. **来源可信吗** —— 脚本要求显式给 ``--sha256``，下载后逐字节校验；
   不给哈希就必须加 ``--insecure-no-hash``，且会明确警告。
2. **放哪** —— 落到 ``models/yolo/plate_det.onnx``，这是引擎唯一会找的路径。
3. **能不能用** —— 这一条最容易踩坑：导出的 YOLO-pose 权重，输出张量的
   **属性列顺序有两种互不兼容的约定**，而它们的属性数完全相同，
   光看形状分不出来。脚本会把模型跑一遍、解出形状、告诉你本引擎
   会解析成什么布局，以及还需要你确认什么。

第 3 条是重点：直接说"下载完成"然后让用户在 ``lpr engines`` 那里看到
"可用"却跑出乱框，比不给工具更糟。

## 用法

列出已知来源与它们的真实许可状态::

    python tools/fetch_yolo_weights.py --list

从 URL 下载（**必须**给哈希）::

    python tools/fetch_yolo_weights.py --url https://…/plate_det.onnx --sha256 <64位十六进制>

从 GitHub Release 取资产::

    python tools/fetch_yolo_weights.py --github owner/repo --asset '*det*.onnx'

只看某个已有权重能不能被解析::

    python tools/fetch_yolo_weights.py --inspect models/yolo/plate_det.onnx

自己训练权重请走 ``tools/ccpd_to_yolo.py`` + ``tools/train_plate_yolo.py``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lpr.engines.yolo_engine import (
    ATTR_ORDER_CLS_FIRST,
    ATTR_ORDER_KPT_FIRST,
    DETECT_MODEL_NAME,
    locate_model_dir,
)

#: 下载超时（秒）
TIMEOUT = 120
#: 下载大小上限（500 MB）—— 防手滑指向一个大文件
MAX_BYTES = 500 * 1024 * 1024


@dataclass(frozen=True)
class Source:
    """一个已知的权重来源。"""

    key: str
    title: str
    license: str
    #: 能否脚本化下载
    auto: bool
    #: 本引擎的解码器是否支持它的输出布局
    supported: bool
    #: 直接下载地址（``auto=False`` 时为空）
    url: str
    #: 期望的 SHA256；未知则留空（此时 ``auto`` 必为 False，不能盲下）
    sha256: str
    #: 输出布局约定（能确认的写死，不能确认的说明如何判定）
    attr_order: str
    note: str


#: 人工核实过的来源清单。
#:
#: **只登记真的查过的东西**：许可从仓库的 license 字段读出来，布局从对方
#: 推理脚本的后处理代码里读出来。没核实过的宁可不写 —— 这个清单的价值
#: 全在于"照着做不会踩坑"，写一条错的就废了。
SOURCES: dict[str, Source] = {
    "we0091234-det": Source(
        key="we0091234-det",
        title="we0091234/Chinese_license_plate_detection_recognition（plate_detect）",
        license="GPL-3.0",
        auto=False,
        supported=True,
        url="",
        sha256="",
        attr_order=ATTR_ORDER_KPT_FIRST,
        note=(
            "与本项目同为 GPL-3.0，许可上可直接组合。仓库里放的是 PyTorch 权重"
            "（weights/plate_detect.pt），ONNX 需要自己用 export.py 导出，或用作者"
            "在 README 里给的网盘链接。\n"
            "        布局已从它的 onnx_infer.py 后处理确认：dets[:,:,4] 取 obj、"
            "dets[:,5:13] 取四角点、dets[:,13:15] 取两类 argmax → "
            "**attr_order 必须设为 kpt_first**，否则框会全错。"
        ),
    ),
    "opencv-zoo-lpd": Source(
        key="opencv-zoo-lpd",
        title="opencv/opencv_zoo license_plate_detection_yunet（YuNet 车牌检测）",
        license="Apache-2.0",
        auto=False,
        supported=False,
        url="",
        sha256="",
        attr_order="",
        note=(
            "许可最干净（Apache-2.0，由 Watrix.ai 贡献，明确说明是针对**中国车牌**"
            "训练的），但**本引擎目前不支持**：它是 YuNet 的多头架构，输出 "
            "``loc`` / ``conf`` / ``iou`` 三个张量，需要按 stride 生成先验框后再解码"
            "（见其 lpd_yunet.py 的 _decode），与 YOLO 系的单张量输出不是一回事。\n"
            "        列在这里是为了让你**别白花时间**：想用它可以，但要先在本引擎里"
            "补一个 YuNet 解码分支。"
        ),
    ),
}


def sha256_file(path: Path) -> str:
    """流式算文件 SHA256（不吃内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, *, expect_sha256: str = "", force: bool = False) -> int:
    """下载到 ``dest``（先写临时文件再原子替换），返回退出码。

    原子替换很重要：中途断网时不能留下一个"看起来存在但残缺"的权重 ——
    引擎只会检查文件是否存在，残缺文件会一路拖到推理时才炸。
    """
    if dest.exists() and not force:
        print(f"✗ {dest} 已存在；确认要覆盖请加 --force", file=sys.stderr)
        return 2

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "lpr-yolo-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            declared = resp.headers.get("Content-Length")
            total = int(declared) if declared and declared.isdigit() else 0
            if total > MAX_BYTES:
                print(
                    f"✗ 文件过大（{total / 1e6:.1f} MB > {MAX_BYTES / 1e6:.0f} MB）",
                    file=sys.stderr,
                )
                return 2

            with tempfile.NamedTemporaryFile(
                dir=dest.parent, prefix=".dl-", suffix=".part", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                written = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_BYTES:
                        tmp.close()
                        tmp_path.unlink(missing_ok=True)
                        print("✗ 下载超过大小上限，已中止", file=sys.stderr)
                        return 2
                    tmp.write(chunk)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"✗ 下载失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"  已下载 {written / 1e6:.2f} MB")

    if expect_sha256:
        actual = sha256_file(tmp_path)
        if actual.lower() != expect_sha256.lower():
            tmp_path.unlink(missing_ok=True)
            print("✗ SHA256 不匹配，已丢弃下载结果", file=sys.stderr)
            print(f"  期望 {expect_sha256.lower()}", file=sys.stderr)
            print(f"  实际 {actual}", file=sys.stderr)
            return 1
        print(f"  ✓ SHA256 校验通过 ({actual[:16]}…)")
    else:
        print("  ! 未提供 --sha256，**无法确认文件完整性**（随时可能被投毒）")

    shutil.move(str(tmp_path), str(dest))
    print(f"  已写入 {dest}")
    return 0


def github_release_assets(repo: str, tag: str | None) -> list[dict]:
    """列出 GitHub Release 的资产（``tag=None`` 取 latest）。"""
    endpoint = f"releases/tags/{tag}" if tag else "releases/latest"
    url = f"https://api.github.com/repos/{repo}/{endpoint}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "lpr-yolo-fetch/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"✗ {repo} 没有 release {'tag ' + tag if tag else '(latest)'}", file=sys.stderr)
        else:
            print(f"✗ GitHub API {exc.code}: {exc.read().decode()[:300]}", file=sys.stderr)
        return []
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"✗ 网络错误: {exc}", file=sys.stderr)
        return []
    return list(data.get("assets") or [])


def inspect_model(path: Path) -> int:
    """跑一次会话，报告输出形状与引擎能解析出的布局。

    这是本脚本最有价值的一步：**只看形状无法确定属性列顺序**，而两种约定
    的属性数又完全相同。所以这里把候选都列出来，并给出判定方法。
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print('需要 onnxruntime 才能检查模型：pip install -e ".[yolo]"', file=sys.stderr)
        return 2

    from lpr.engines.yolo_engine import resolve_output_spec

    if not path.is_file():
        print(f"✗ 文件不存在: {path}", file=sys.stderr)
        return 2

    print(f"文件   {path}")
    print(f"大小   {path.stat().st_size / 1e6:.2f} MB")
    print(f"SHA256 {sha256_file(path)}")

    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        print(f"✗ 会话建立失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\n输入:")
    for i in sess.get_inputs():
        print(f"  {i.name:14} {i.shape} {i.type}")
    print("输出:")
    for o in sess.get_outputs():
        print(f"  {o.name:14} {o.shape} {o.type}")

    if len(sess.get_outputs()) != 1:
        print(
            f"\n! 该模型有 {len(sess.get_outputs())} 个输出。本引擎按**单个**检测张量"
            "解码；多头架构（如 YuNet 的 loc/conf/iou）需要另写解码分支。"
        )

    print("\n本引擎能解析出的布局:")
    found = False
    for order, label in (
        (ATTR_ORDER_CLS_FIRST, "cls_first（Ultralytics：box+[obj]+cls+kpts）"),
        (ATTR_ORDER_KPT_FIRST, "kpt_first（we0091234：box+obj+kpts+cls）"),
    ):
        for out in sess.get_outputs():
            try:
                spec = resolve_output_spec(out.shape, num_classes=2, attr_order=order)
            except (ValueError, TypeError):
                continue
            found = True
            print(f"  {out.name} + {label}")
            print(f"      → {spec}")
    if not found:
        print("  ✗ 没有任何组合能解析 —— 输出形状与本引擎支持的车牌 pose 布局不符")
        print("    （支持：4+nc、4+1+nc、4+nc+8、4+1+nc+8、4+nc+12 等）")
        return 1

    print(
        "\n下一步：确认**属性列顺序**（形状区分不了，属性数完全相同）。"
        "\n  · 用 Ultralytics yolo export 出来的 → cls_first（默认值，无需设置）"
        '\n  · 用 we0091234 系权重或按其后处理写的 → 在 YoloEngine(attr_order="kpt_first")'
        "\n  判定方法：去看训练/导出那一侧的推理脚本怎么切列 "
        "（例如 obj 取第 4 位还是第 5 位）。"
    )
    return 0


def list_sources() -> int:
    print("已知来源（人工核实过许可与输出布局）：\n")
    for src in SOURCES.values():
        flag = "可脚本下载" if src.auto else "需手动获取"
        support = "本引擎支持" if src.supported else "**本引擎暂不支持**"
        print(f"[{src.key}] {src.title}")
        print(f"    许可: {src.license} | {flag} | {support}")
        if src.attr_order:
            print(f"    列顺序: {src.attr_order}")
        print(f"    {src.note}\n")
    print(
        "想自己训练：python tools/ccpd_to_yolo.py --help && python tools/train_plate_yolo.py --help"
    )
    print(
        "只想先验证管道：python tools/make_stub_onnx.py --out models/yolo/plate_det.onnx --selftest"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="获取/校验 yolo 引擎的 ONNX 权重，并检查能否被本引擎解析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list", action="store_true", help="列出已知来源及其许可/支持情况")
    ap.add_argument("--url", help="下载地址")
    ap.add_argument(
        "--sha256", default="", help="期望的 SHA256（强烈建议给；不给需 --insecure-no-hash）"
    )
    ap.add_argument(
        "--insecure-no-hash",
        action="store_true",
        help="允许在没有哈希的情况下下载（有被投毒风险，仅用于可信内网）",
    )
    ap.add_argument("--github", help="从 GitHub Release 取资产，格式 owner/repo")
    ap.add_argument("--tag", help="指定 release tag（默认 latest）")
    ap.add_argument("--asset", default="*.onnx", help="Release 资产名通配（默认 *.onnx）")
    ap.add_argument("--inspect", metavar="PATH", help="只检查一个已有 ONNX 模型，不下载")
    ap.add_argument("--out", help="输出路径（默认 models/yolo/plate_det.onnx）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    args = ap.parse_args()

    if args.list:
        return list_sources()

    if args.inspect:
        return inspect_model(Path(args.inspect))

    dest = Path(args.out) if args.out else locate_model_dir() / DETECT_MODEL_NAME

    if args.github:
        assets = github_release_assets(args.github, args.tag)
        if not assets:
            return 1
        matches = [
            a for a in assets if Path(a["name"]).match(args.asset) or args.asset == a["name"]
        ]
        if not matches:
            print(
                f"✗ {args.github} 的 release 里没有匹配 {args.asset!r} 的资产。有：",
                file=sys.stderr,
            )
            for a in assets:
                print(f"    {a['name']}  ({a['size'] / 1e6:.2f} MB)", file=sys.stderr)
            return 1
        if len(matches) > 1:
            print(f"! 匹配到多个资产，取第一个：{[a['name'] for a in matches]}", file=sys.stderr)
        chosen = matches[0]
        print(f"选中资产 {chosen['name']} ({chosen['size'] / 1e6:.2f} MB)")
        # Release 资产的 digest 字段（新 API）可直接当哈希用；没有就要求显式给
        digest = (chosen.get("digest") or "").removeprefix("sha256:")
        expect = args.sha256 or digest
        if not expect and not args.insecure_no_hash:
            print(
                "✗ 该资产的 digest 不可用，请用 --sha256 提供期望哈希，"
                "或加 --insecure-no-hash 承担风险",
                file=sys.stderr,
            )
            return 2
        code = download(
            chosen["browser_download_url"], dest, expect_sha256=expect, force=args.force
        )
        return code if code else inspect_model(dest)

    if args.url:
        if not args.sha256 and not args.insecure_no_hash:
            print(
                "✗ 从任意 URL 下载**必须**提供 --sha256，否则无法确认拿到的是不是你要的文件。\n"
                "  确实要跳过校验请加 --insecure-no-hash。",
                file=sys.stderr,
            )
            return 2
        code = download(args.url, dest, expect_sha256=args.sha256, force=args.force)
        return code if code else inspect_model(dest)

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
