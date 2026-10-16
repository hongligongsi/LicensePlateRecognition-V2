"""``tools/ccpd_to_yolo.py`` 与 ``tools/fetch_yolo_weights.py`` 的测试。

重点在「会静默毁掉训练」的两处：

* **四角点顺序**：CCPD 是「右下→左下→左上→右上」，YOLO-pose 要
  「左上→右上→右下→左下」。写错不报错，只会训出一个框歪着的模型。
* **``flip_idx``**：水平翻转后关键点必须重排；漏写同样不报错。

所以这里把置换与翻转映射都用**坐标级断言**钉死，而不是只测"文件生成了"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import ccpd_to_yolo  # noqa: E402
from ccpd_labels import CcpdAnnotation, parse  # noqa: E402
from ccpd_to_yolo import (  # noqa: E402
    CCPD_TO_YOLO_CORNER_ORDER,
    CLASS_NAMES,
    DOUBLE_ASPECT_THRESHOLD,
    FLIP_IDX,
    annotate,
    is_val,
    render_data_yaml,
)


def make_ccpd_name(
    corners: list[tuple[int, int]],
    *,
    area: int = 95,
    tilt: str = "11_3",
    brightness: int = 37,
    blur: int = 15,
) -> str:
    """按 CCPD 命名规范拼一个文件名。

    CCPD 的字段顺序是 面积占比-倾角-bbox-四角点-号码-亮度-模糊度，
    四角点顺序为 **右下 / 左下 / 左上 / 右上**。
    """
    pts = "_".join(f"{x}&{y}" for x, y in corners)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    bbox = f"{min(xs)}&{min(ys)}_{max(xs)}&{max(ys)}"
    return f"{area}-{tilt}-{bbox}-{pts}-0_0_22_27_27_33_16-{brightness}-{blur}.jpg"


def annotation_for(corners: list[tuple[int, int]], path: Path | None = None) -> CcpdAnnotation:
    name = make_ccpd_name(corners)
    ann = parse(name)
    assert ann is not None
    if path is None:
        return ann
    return CcpdAnnotation(
        path=path,
        text=ann.text,
        bbox=ann.bbox,
        corners=ann.corners,
        tilt=ann.tilt,
        area_ratio=ann.area_ratio,
        brightness=ann.brightness,
        blurriness=ann.blurriness,
    )


#: 一个贴近真实的 CCPD 样本：车牌略侧倾，宽约 210、高约 75（单行牌）
TILTED_CORNERS = [(386, 473), (177, 454), (154, 383), (363, 402)]
#: 双层牌形状：宽 200、高 100 → 比值 0.5
DOUBLE_CORNERS = [(200, 100), (0, 100), (0, 0), (200, 0)]

#: 新版导出的真实文件名（``datasets/ccpd_labeled/_labels.json`` 里的 ``orig``）。
#: 与老格式的三处差别：坐标用 ``,`` 分隔、面积是长整数、模糊度带 ``_ccpd_base_*`` 尾巴。
NEW_FORMAT_NAME = (
    "0189655172414-91_85-233,591_457,681"
    "-463,678_245,663_232,585_450,600-0_3_15_27_28_29_30-134-129_ccpd_base_005128.jpg"
)


class TestCommaSeparatorFormat:
    """新版导出用 ``,`` 当坐标分隔符。

    这是真踩过的坑：``parse`` 早先只认 ``&``，于是对**整个新格式数据集**
    静默返回 ``None`` —— 评测脚本拿到 0/100 的解析率而不报错。所以这里既测
    "两种写法等价"，也测"逗号格式下的 bbox 真的取到了值"。
    """

    def test_comma_format_parses(self):
        ann = parse(NEW_FORMAT_NAME)
        assert ann is not None

    def test_comma_and_ampersand_forms_agree(self):
        pts = [(386, 473), (177, 454), (154, 383), (363, 402)]
        old = parse(make_ccpd_name(pts))
        new = parse(make_ccpd_name(pts).replace("&", ","))
        assert old is not None and new is not None
        assert old.bbox == new.bbox
        assert old.corners == new.corners

    def test_bbox_is_split_on_comma(self):
        ann = parse(NEW_FORMAT_NAME)
        assert ann is not None
        assert ann.bbox == (233, 591, 457, 681)

    def test_corners_are_split_on_comma(self):
        ann = parse(NEW_FORMAT_NAME)
        assert ann is not None
        assert ann.corners == [(463, 678), (245, 663), (232, 585), (450, 600)]

    def test_trailing_suffix_on_blur_field_does_not_break_parsing(self):
        """模糊度字段是 ``129_ccpd_base_005128`` —— 必须只取前导整数。"""
        ann = parse(NEW_FORMAT_NAME)
        assert ann is not None
        assert ann.blurriness == 129
        assert ann.brightness == 134

    def test_unknown_separator_is_rejected_not_silently_misparsed(self):
        assert parse(NEW_FORMAT_NAME.replace(",", ":")) is None


class TestCornerOrdering:
    """四角点顺序 —— 错了不报错，只毁模型。"""

    def test_reorder_is_rb_lb_lt_rt_to_lt_rt_rb_lb(self):
        """置换必须是 [2, 3, 0, 1]。"""
        assert CCPD_TO_YOLO_CORNER_ORDER == (2, 3, 0, 1)

    def test_points_are_emitted_in_lt_rt_rb_lb_order(self):
        """输出的 4 个点必须依次是 左上 / 右上 / 右下 / 左下。

        用一组**四个角都有辨识度**的坐标，逐个核对：
        左上 = (154,383)、右上 = (363,402)、右下 = (386,473)、左下 = (177,454)。
        """
        ann = annotation_for(TILTED_CORNERS)
        pose = annotate(ann, 800, 600)
        assert pose is not None

        w, h = 800.0, 600.0
        assert pose.points[0] == pytest.approx((154 / w, 383 / h), abs=1e-6)  # 左上
        assert pose.points[1] == pytest.approx((363 / w, 402 / h), abs=1e-6)  # 右上
        assert pose.points[2] == pytest.approx((386 / w, 473 / h), abs=1e-6)  # 右下
        assert pose.points[3] == pytest.approx((177 / w, 454 / h), abs=1e-6)  # 左下

    def test_flip_idx_matches_horizontal_mirror(self):
        """``flip_idx`` 必须真的对应水平翻转的置换。

        顺序为 [LT, RT, RB, LB]。图像水平翻转后，屏幕上"左上"位置出现的
        其实是原来的"右上"点。所以翻转后的关键点序列应当是
        ``[mirror(RT), mirror(LT), mirror(LB), mirror(RB)]``，
        也就是按 ``[1, 0, 3, 2]`` 取原序列后再镜像 —— 与 ``FLIP_IDX`` 一致。

        漏写或写错这个映射，翻转增广会把角点学乱，而且**训练不会报错**。
        """
        ann = annotation_for(TILTED_CORNERS)
        pose = annotate(ann, 800, 600)
        assert pose is not None

        mirrored = tuple((1.0 - x, y) for x, y in pose.points)
        flipped = tuple(mirrored[i] for i in FLIP_IDX)

        # 逐位核对：翻转后的第 0 位来自原第 1 位（右上），第 1 位来自原第 0 位（左上）
        assert flipped[0] == pytest.approx(mirrored[1])
        assert flipped[1] == pytest.approx(mirrored[0])
        assert flipped[2] == pytest.approx(mirrored[3])
        assert flipped[3] == pytest.approx(mirrored[2])

        # 语义自洽性：索引 0 必须始终是"左上"（x 小于索引 1 的"右上"）。
        # 翻转后原来的右上跑到左边成了新的左上，所以这里应当是 <。
        assert flipped[0][0] < flipped[1][0]
        assert flipped[3][0] < flipped[2][0]

    def test_a_naive_clockwise_shift_would_be_wrong(self):
        """反例：单纯循环移位不是水平翻转的置换。

        这条用例的价值在于证明上一条不是"任何置换都能过" —— 如果写的不是
        镜像映射，坐标检查必须挂掉。
        """
        ann = annotation_for(TILTED_CORNERS)
        pose = annotate(ann, 800, 600)
        assert pose is not None

        mirrored = tuple((1.0 - x, y) for x, y in pose.points)
        wrong = tuple(mirrored[i] for i in (1, 2, 3, 0))  # 顺时针挪一位
        correct = tuple(mirrored[i] for i in FLIP_IDX)
        assert wrong != correct


class TestLayoutClassification:
    """单行 / 双层判定。"""

    def test_single_line_plate_is_class_zero(self):
        ann = annotation_for(TILTED_CORNERS)
        pose = annotate(ann, 800, 600)
        assert pose is not None
        assert pose.class_id == 0

    def test_double_line_plate_is_class_one(self):
        ann = annotation_for(DOUBLE_CORNERS)
        pose = annotate(ann, 800, 600)
        assert pose is not None
        assert pose.class_id == 1

    def test_threshold_sits_between_the_two_real_ratios(self):
        """阈值必须落在真实牌型的比值之间：单行 0.318，双层 0.50。"""
        assert 0.318 < DOUBLE_ASPECT_THRESHOLD < 0.50

    def test_class_ids_survive_a_round_trip_through_the_label_line(self):
        """类别号是训练目标，写进文本后必须能被原样读回。"""
        ann = annotation_for(DOUBLE_CORNERS)
        pose = annotate(ann, 800, 600)
        assert pose is not None
        line = pose.to_line()
        assert len(line.split()) == 13  # 1 类别 + 4 框 + 8 关键点
        assert int(line.split()[0]) == 1


class TestDegenerateAnnotations:
    """坏标注必须被丢掉，而不是喂给训练。"""

    def test_collinear_corners_are_rejected(self):
        """四点几乎共线（高度趋零）时不能产出一条"零高度车牌"。"""
        flat = [(200, 5), (0, 5), (0, 4), (200, 4)]
        ann = annotation_for(flat)
        assert annotate(ann, 800, 600) is None

    def test_tiny_region_is_rejected(self):
        tiny = [(3, 3), (0, 3), (0, 2), (3, 2)]
        ann = annotation_for(tiny)
        assert annotate(ann, 800, 600) is None

    def test_wrong_number_of_corners_is_rejected(self):
        ann = annotation_for(TILTED_CORNERS)
        broken = CcpdAnnotation(
            path=ann.path,
            text=ann.text,
            bbox=ann.bbox,
            corners=ann.corners[:3],
            tilt=ann.tilt,
            area_ratio=ann.area_ratio,
            brightness=ann.brightness,
            blurriness=ann.blurriness,
        )
        assert annotate(broken, 800, 600) is None

    def test_degenerate_image_size_is_rejected(self):
        ann = annotation_for(TILTED_CORNERS)
        assert annotate(ann, 0, 600) is None


class TestValSplit:
    """验证集切分必须可复现 —— 否则两次训练的指标没法比。"""

    def test_same_name_always_lands_in_the_same_split(self):
        assert is_val("abc.jpg", 0.3) == is_val("abc.jpg", 0.3)

    def test_ratio_bounds_are_honoured(self):
        assert is_val("anything.jpg", 0.0) is False
        assert is_val("anything.jpg", 1.0) is True

    def test_split_is_roughly_proportional(self):
        names = [f"{i}.jpg" for i in range(2000)]
        share = sum(is_val(n, 0.2) for n in names) / len(names)
        assert 0.15 < share < 0.25


class TestDataYaml:
    """data.yaml 的每一行都影响训练正确性。"""

    def test_contains_kpt_shape(self):
        assert "kpt_shape: [4, 2]" in render_data_yaml(Path("/tmp/ds"))

    def test_contains_flip_idx(self):
        """漏了 flip_idx，翻转增广会把关键点学乱且不报错。"""
        text = render_data_yaml(Path("/tmp/ds"))
        assert "flip_idx: [1, 0, 3, 2]" in text

    def test_class_names_match_engine_convention(self):
        """类 1 必须是双层牌 —— 与 YoloEngine.double_class_ids 默认值一致。"""
        assert CLASS_NAMES == ("single", "double")
        text = render_data_yaml(Path("/tmp/ds"))
        assert "0: single" in text
        assert "1: double" in text

    def test_path_is_absolute_posix(self):
        text = render_data_yaml(Path("/tmp/ds"))
        assert "path: /tmp/ds" in text


class TestEndToEndConversion:
    """真的跑一遍 CLI：建图 → 转换 → 检查产物。"""

    @pytest.fixture()
    def ccpd_dir(self, tmp_path: Path) -> Path:
        """造一个含 4 张图的小 CCPD 目录（2 单行 + 2 双层）。"""
        cv2 = pytest.importorskip("cv2")
        src = tmp_path / "ccpd"
        src.mkdir()
        specs = [
            (TILTED_CORNERS, 800),
            (DOUBLE_CORNERS, 400),
            ([(300, 220), (100, 210), (95, 150), (295, 160)], 600),
            ([(400, 400), (200, 400), (200, 300), (400, 300)], 500),
        ]
        for corners, size in specs:
            name = make_ccpd_name(corners)
            img = np.full((size, size, 3), 60, np.uint8)
            cv2.imwrite(str(src / name), img)
        return src

    def test_produces_images_labels_and_config(self, ccpd_dir: Path, tmp_path: Path):
        import sys as _sys

        out = tmp_path / "pose"
        argv_backup = _sys.argv
        try:
            _sys.argv = [
                "ccpd_to_yolo.py",
                "--ccpd",
                str(ccpd_dir),
                "--out",
                str(out),
                "--mode",
                "copy",
                "--val-ratio",
                "0.0",
            ]
            rc = ccpd_to_yolo.main()
        finally:
            _sys.argv = argv_backup

        assert rc == 0
        assert (out / "data.yaml").is_file()
        labels = sorted((out / "labels" / "train").glob("*.txt"))
        assert len(labels) == 4, f"应产出 4 条标注，实际 {len(labels)}"
        assert len(list((out / "images" / "train").glob("*.jpg"))) == 4

        for label in labels:
            line = label.read_text(encoding="utf-8").strip()
            fields = line.split()
            assert len(fields) == 13
            cls = int(fields[0])
            assert cls in (0, 1)
            # 归一化坐标必须落在 [0, 1]
            for v in (float(x) for x in fields[1:]):
                assert 0.0 <= v <= 1.0

    def test_dry_run_writes_nothing(self, ccpd_dir: Path, tmp_path: Path):
        import sys as _sys

        out = tmp_path / "pose_dry"
        argv_backup = _sys.argv
        try:
            _sys.argv = [
                "ccpd_to_yolo.py",
                "--ccpd",
                str(ccpd_dir),
                "--out",
                str(out),
                "--dry-run",
            ]
            rc = ccpd_to_yolo.main()
        finally:
            _sys.argv = argv_backup
        assert rc == 0
        assert not out.exists()

    def test_directory_without_labels_is_reported(self, tmp_path: Path):
        """ModelScope 镜像那种 sample_N.jpg 没有标注，必须明确报错而不是静默产出空集。"""
        import sys as _sys

        src = tmp_path / "mirror"
        src.mkdir()
        (src / "sample_1.jpg").write_bytes(b"not really an image")

        argv_backup = _sys.argv
        try:
            _sys.argv = [
                "ccpd_to_yolo.py",
                "--ccpd",
                str(src),
                "--out",
                str(tmp_path / "o"),
                "--dry-run",
            ]
            rc = ccpd_to_yolo.main()
        finally:
            _sys.argv = argv_backup
        assert rc == 2

    def test_missing_directory_is_reported(self, tmp_path: Path):
        import sys as _sys

        argv_backup = _sys.argv
        try:
            _sys.argv = [
                "ccpd_to_yolo.py",
                "--ccpd",
                str(tmp_path / "nope"),
                "--dry-run",
            ]
            rc = ccpd_to_yolo.main()
        finally:
            _sys.argv = argv_backup
        assert rc == 2


class TestFetchWeightsTool:
    """权重获取工具：哈希校验与来源登记。"""

    def test_registry_entries_have_required_metadata(self):
        from fetch_yolo_weights import SOURCES

        assert SOURCES
        for key, src in SOURCES.items():
            assert src.key == key
            assert src.title and src.license
            # auto=True 的条目必须给出 URL 与哈希，否则就是"看着能自动下、
            # 实际会失败"的假承诺
            if src.auto:
                assert src.url, f"{key} 标了 auto 却没给 url"
                assert src.sha256, f"{key} 标了 auto 却没给 sha256"

    def test_we0091234_entry_uses_kpt_first(self):
        """已核实的布局必须写对 —— 写错会让用户拿到全错的框。"""
        from fetch_yolo_weights import SOURCES

        src = SOURCES["we0091234-det"]
        assert src.attr_order == "kpt_first"
        assert src.supported is True
        assert src.license == "GPL-3.0"

    def test_yunet_entry_is_marked_unsupported(self):
        """YuNet 是不支持的多头架构，必须显式标出来，免得用户白花时间。"""
        from fetch_yolo_weights import SOURCES

        src = SOURCES["opencv-zoo-lpd"]
        assert src.supported is False
        assert src.license == "Apache-2.0"

    def test_sha256_file_matches_hashlib(self, tmp_path: Path):
        import hashlib

        from fetch_yolo_weights import sha256_file

        data = b"license plate weights payload"
        p = tmp_path / "w.onnx"
        p.write_bytes(data)
        assert sha256_file(p) == hashlib.sha256(data).hexdigest()

    def test_download_refuses_existing_file_without_force(self, tmp_path: Path):
        from fetch_yolo_weights import download

        dest = tmp_path / "plate_det.onnx"
        dest.write_bytes(b"existing")
        # 已存在且未给 --force → 直接拒绝，不联网
        assert download("https://example.invalid/x.onnx", dest, force=False) == 2
        assert dest.read_bytes() == b"existing"

    def test_url_requires_hash(self, tmp_path: Path):
        """从任意 URL 下载必须给哈希 —— 这个闸门是防投毒的关键。"""
        import sys as _sys

        from fetch_yolo_weights import main as fetch_main

        argv_backup = _sys.argv
        try:
            _sys.argv = [
                "fetch_yolo_weights.py",
                "--url",
                "https://example.invalid/x.onnx",
                "--out",
                str(tmp_path / "plate_det.onnx"),
            ]
            rc = fetch_main()
        finally:
            _sys.argv = argv_backup
        assert rc == 2
