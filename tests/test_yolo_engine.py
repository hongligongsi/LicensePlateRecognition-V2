"""检测识别引擎（``yolo``）的测试。

分两层：

* **纯函数层** —— ``resolve_output_spec`` / ``decode_predictions`` / ``letterbox``
  / ``nms`` / ``ctc_decode``。这些是整条链路的算术核心，用合成张量直接钉住，
  不需要任何权重。
* **端到端层** —— 用 ``tools/make_stub_onnx.py`` 生成**常量输出的桩 ONNX
  模型**，喂给真正的 ``onnxruntime`` 会话，验证 会话建立 → 输入名 →
  letterbox → 解码 → NMS → 逆 letterbox → 角点矫正 的坐标自洽性。

端到端层刻意**不依赖任何第三方权重**：真权重有体积与许可问题（不进仓库），
但"管道通不通"必须能无条件验证。桩模型的框是已知常量，所以能把解出来的
坐标和手算结果逐位对齐 —— 这比"跑通就行"强得多。

设计意图（为什么这些断言值得存在）：

``yolo`` 引擎的权重不进仓库，用户装完 extras 后**大概率**看不到它出现在
下拉框里。此时唯一能帮到用户的就是「一条说明去哪拿权重的报错」。
v2.3.5 在 ``start.bat`` 上吃过反面教训（静默失败 → 用户以为代码坏了），
所以这里把「错误提示必须包含路径与安装命令」钉成测试。
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from lpr.engines import YoloEngine, available_engines, create_engine, describe_engines  # noqa: E402
from lpr.engines.yolo_engine import (  # noqa: E402
    ATTR_ORDER_CLS_FIRST,
    ATTR_ORDER_KPT_FIRST,
    DETECT_MODEL_NAME,
    RECOGNIZE_MODEL_NAME,
    Detection,
    OutputSpec,
    _to_xywh,
    ctc_decode,
    decode_predictions,
    letterbox,
    locate_model_dir,
    nms,
    resolve_output_spec,
)
from lpr.types import PlateLayout  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]

#: 桩模型输入图尺寸（与 ``YoloEngine.input_size`` 默认值一致）
STUB_INPUT = 640
#: 桩模型里写死的框（letterbox 坐标系）：cx, cy, w, h
STUB_BOX = (320.0, 320.0, 200.0, 64.0)


class TestRegistration:
    """注册与发现 —— 界面/CLI 靠这两条自动看到新引擎。"""

    def test_engine_is_registered(self):
        assert "yolo" in available_engines()

    def test_registry_order_includes_all_three(self):
        # 三个内置引擎都要在，且顺序稳定（下拉框按名字排序）
        assert available_engines() == ["cv", "deep", "yolo"]

    def test_describe_engines_carries_display_name_and_reason(self):
        info = next(e for e in describe_engines() if e["name"] == "yolo")
        assert info["display_name"] == "检测识别引擎"
        assert info["description"]
        # 字段齐备，界面才能渲染「不可用 + 原因」而不崩
        assert set(info) >= {"name", "display_name", "description", "available", "reason"}

    def test_create_engine_by_name(self):
        engine = create_engine("yolo", None)
        assert isinstance(engine, YoloEngine)
        assert engine.name == "yolo"

    def test_available_engines_are_all_instantiable(self):
        """每个注册的引擎都要能被 create_engine 造出来（不加载模型）。"""
        for name in available_engines():
            engine = create_engine(name, None)
            assert engine.name == name

    def test_engine_is_selectable_from_config_and_cli(self):
        """``yolo`` 必须是**可选**的引擎，不能只在 ``lpr engines`` 里露个脸。

        此前白名单硬编码成 ``("cv", "deep", "cascade")`` 且抄了三份，导致
        权重装好了也选不动 —— config 校验拒绝、CLI argparse 直接报非法值。
        """
        from lpr.cli import _make_parser
        from lpr.config import ENGINE_CHOICES, EngineConfig

        assert "yolo" in ENGINE_CHOICES
        EngineConfig(default="yolo").validate()

        ns = _make_parser().parse_args(["predict", "x.jpg", "--engine", "yolo"])
        assert ns.engine == "yolo"


class TestAvailabilityProbe:
    """可用性探测：onnxruntime 与权重**同时**具备才算可用。"""

    def test_unavailable_without_weights(self):
        """默认权重缺失时不可用（models/yolo 下没有 onnx）。

        提示必须可行动 —— 给出确切路径或安装命令，不能只说「不可用」。
        """
        ok, reason = YoloEngine.is_available()
        if ok:  # 已放入权重（例如自定义部署）则不适用本断言
            return
        assert str(locate_model_dir() / DETECT_MODEL_NAME) in reason or "onnxruntime" in reason

    def test_probe_reports_the_expected_path(self, tmp_path):
        """把 models_dir 指到空目录时，提示里必须出现那个具体路径。"""
        engine = YoloEngine(models_dir=tmp_path)
        assert engine.detect_model_path == tmp_path / DETECT_MODEL_NAME
        assert engine.recognize_model_path == tmp_path / RECOGNIZE_MODEL_NAME

    def test_load_raises_engine_unavailable_when_model_missing(self, tmp_path):
        """权重缺失时必须抛 EngineUnavailable，而不是 FileNotFoundError。

        界面层靠这个异常类型给出「请安装 XXX」的提示（base.EngineUnavailable
        的契约），抛别的类型会被当成真 bug 弹栈。
        """
        from lpr.engines.base import EngineUnavailable

        engine = YoloEngine(models_dir=tmp_path)
        with pytest.raises(EngineUnavailable) as exc:
            engine.load()
        assert "plate_det.onnx" in str(exc.value)

    def test_instance_check_honours_custom_models_dir(self, tmp_path):
        """实例的可用性判断必须看**自己的** models_dir。

        类方法 ``is_available()`` 只能看默认目录（``describe_engines`` 要求
        它是类方法）。若 ``load()`` 也用它，把权重放在别处的用户就会遇到
        "文件明明在、却报找不到" —— 这个 bug 真的出现过。
        """
        engine = YoloEngine(models_dir=tmp_path)
        ok, reason = engine._check_available()
        if ok:  # onnxruntime 都没装时跳过
            pytest.skip("onnxruntime 不可用")
        assert str(tmp_path / DETECT_MODEL_NAME) in reason


class TestPathResolution:
    """路径解析 —— 权重放错目录是最常见的用户错误。"""

    def test_default_dir_is_models_yolo_under_project_root(self):
        assert locate_model_dir() == PROJECT / "models" / "yolo"

    def test_explicit_models_dir_wins(self, tmp_path):
        assert locate_model_dir(tmp_path) == tmp_path

    def test_models_dir_is_none_by_default(self):
        assert YoloEngine().models_dir == PROJECT / "models" / "yolo"


class TestDefaultsAndConfig:
    """默认参数 —— 改这些会直接影响与导出权重的一致性。"""

    def test_input_size_default_matches_yolo_convention(self):
        assert YoloEngine().input_size == 640

    def test_thresholds_are_sane(self):
        engine = YoloEngine()
        assert 0.0 < engine.conf_threshold < 1.0
        assert 0.0 < engine.iou_threshold < 1.0

    def test_config_kwarg_is_accepted(self):
        """create_engine 会传 config=...，签名必须收得下（否则落到 TypeError 分支）。"""
        engine = YoloEngine(config=None, models_dir=None)
        assert engine.config is None

    def test_default_attr_order_is_cls_first(self):
        """默认列顺序必须与 Ultralytics 一致（主流导出）。"""
        assert YoloEngine().attr_order == ATTR_ORDER_CLS_FIRST


class TestDetectionDataclass:
    """Detection 是检测后处理的输出契约，尺寸属性要稳。"""

    def test_width_height_from_box(self):
        det = Detection(
            corners=np.zeros((4, 2), dtype=np.float32),
            box=(10.0, 20.0, 110.0, 50.0),
            layout=PlateLayout.SINGLE,
            score=0.9,
        )
        assert det.width == 100.0
        assert det.height == 30.0

    def test_corners_are_float32_quad(self):
        det = Detection(
            corners=np.zeros((4, 2), dtype=np.float32),
            box=(0.0, 0.0, 10.0, 5.0),
            layout=PlateLayout.SINGLE,
            score=0.5,
        )
        assert det.corners.shape == (4, 2)
        assert det.corners.dtype == np.float32


# ============================================================ 输出布局解析


class TestResolveOutputSpec:
    """形状 → 语义布局。接错权重时这里是第一道也是最有用的一道闸门。"""

    @pytest.mark.parametrize(
        ("shape", "nc", "expect"),
        [
            # Ultralytics v8-pose：(1, 4+nc+8, N)
            ((1, 14, 8400), 2, OutputSpec(True, False, 4, 2, 2, ATTR_ORDER_CLS_FIRST)),
            # Ultralytics v5-pose：(1, N, 4+1+nc+8)
            ((1, 25200, 15), 2, OutputSpec(False, True, 4, 2, 2, ATTR_ORDER_CLS_FIRST)),
            # 纯检测 v8：(1, 4+nc, N)
            ((1, 6, 8400), 2, OutputSpec(True, False, 0, 2, 2, ATTR_ORDER_CLS_FIRST)),
            # 纯检测 v5：(1, N, 4+1+nc)
            ((1, 25200, 7), 2, OutputSpec(False, True, 0, 2, 2, ATTR_ORDER_CLS_FIRST)),
            # 单类多类都成立：nc=1 → v8-pose attr=13
            ((1, 13, 100), 1, OutputSpec(True, False, 4, 2, 1, ATTR_ORDER_CLS_FIRST)),
            # 角点带置信度（kpt_dim=3）：nc=2 → 4+2+12=18
            ((1, 18, 100), 2, OutputSpec(True, False, 4, 3, 2, ATTR_ORDER_CLS_FIRST)),
        ],
    )
    def test_resolves_supported_layouts(self, shape, nc, expect):
        assert resolve_output_spec(shape, num_classes=nc) == expect

    def test_batch_dimension_is_ignored(self):
        """有无 batch 维都要能解析（某些导出会省掉它）。"""
        with_batch = resolve_output_spec((1, 14, 8400), num_classes=2)
        without = resolve_output_spec((14, 8400), num_classes=2)
        assert with_batch == without

    def test_orientation_is_inferred_from_which_axis_is_smaller(self):
        """属性数远小于锚点数，据此定朝向，不靠猜。"""
        assert resolve_output_spec((1, 14, 8400), num_classes=2).channels_first is True
        assert resolve_output_spec((1, 8400, 14), num_classes=2).channels_first is False

    def test_unresolvable_attribute_count_reports_shape_and_attempts(self):
        """解析不出来时必须报出形状与试过的假设 —— 否则用户无从下手。"""
        with pytest.raises(ValueError) as exc:
            resolve_output_spec((1, 999, 8400), num_classes=2)
        msg = str(exc.value)
        assert "999" in msg and "C×N" in msg

    def test_non_2d_output_is_rejected(self):
        with pytest.raises(ValueError):
            resolve_output_spec((1, 3, 640, 640), num_classes=2)

    def test_unknown_attr_order_is_rejected(self):
        with pytest.raises(ValueError):
            resolve_output_spec((1, 14, 8400), num_classes=2, attr_order="banana")

    def test_attr_count_is_order_independent(self):
        """两种列顺序的属性数完全相同 —— 所以形状区分不出来，必须显式指定。"""
        a = OutputSpec(False, True, 4, 2, 2, ATTR_ORDER_CLS_FIRST)
        b = OutputSpec(False, True, 4, 2, 2, ATTR_ORDER_KPT_FIRST)
        assert a.attr_count == b.attr_count == 15
        # 但列位置不同：这正是需要 attr_order 的原因
        assert a.class_offset != b.class_offset
        assert a.points_offset != b.points_offset


# ============================================================ 解码


def _v5_row(box, obj, corners, cls):
    """按 ``box + obj + cls + kpts`` 拼一条属性向量。"""
    return [*box, obj, *cls, *corners]


def _kpt_first_row(box, obj, corners, cls):
    """按 ``box + obj + kpts + cls`` 拼一条属性向量。"""
    return [*box, obj, *corners, *cls]


class TestDecodePredictions:
    """张量 → 框/分数/类别/角点。"""

    def _pred(self, rows: list[list[float]]) -> np.ndarray:
        """拼成 ``(1, N, attr)``。

        **至少补到 2 个锚点**：单个锚点时形状是 ``(1, 1, attr)``，去掉
        batch 维只剩一维，无法判断哪个轴是属性数 —— 那属于非法输入而非
        "朝向不明"。
        """
        while len(rows) < 2:
            rows = [*rows, [0.0] * len(rows[0])]
        return np.asarray(rows, dtype=np.float32)[None, ...]

    def test_anchors_first_slicing(self):
        corners = [70, 30, 130, 30, 130, 70, 70, 70]
        pred = self._pred([_v5_row([100, 50, 60, 20], 0.9, corners, [0.1, 0.95])])
        spec = resolve_output_spec(pred.shape, num_classes=2)
        boxes, _scores, ids, points = decode_predictions(pred, spec)

        assert boxes[0] == pytest.approx([70, 40, 130, 60])
        assert ids[0] == 1
        assert points is not None and points[0].tolist() == [
            [70, 30],
            [130, 30],
            [130, 70],
            [70, 70],
        ]

    def test_confidence_includes_objectness_when_present(self):
        """有 obj 时 conf = obj × max(cls)：0.9 × 0.95 = 0.855。"""
        corners = [0, 0, 1, 0, 1, 1, 0, 1]
        pred = self._pred([_v5_row([10, 10, 8, 4], 0.9, corners, [0.1, 0.95])])
        _, scores, _, _ = decode_predictions(pred, resolve_output_spec(pred.shape, num_classes=2))
        assert float(scores[0]) == pytest.approx(0.855, abs=1e-5)

    def test_confidence_is_max_class_when_no_objectness(self):
        """无 obj 时 conf = max(cls)，不能凭空乘一个 1。"""
        pred = self._pred([[10, 10, 8, 4, 0.1, 0.95, 0, 0, 1, 0, 1, 1, 0, 1]])
        _, scores, _, _ = decode_predictions(pred, resolve_output_spec(pred.shape, num_classes=2))
        assert float(scores[0]) == pytest.approx(0.95, abs=1e-5)

    def test_kpt_first_slicing_differs_from_cls_first(self):
        """同样 15 个属性，列顺序说错了就会把角点当类别读。

        这条用例把两者的**差异**钉住：同一串数按不同约定解读，得到的
        类别与角点必须不同 —— 否则说明切片逻辑其实是同一个（那就白写了）。
        """
        corners = [70, 30, 130, 30, 130, 70, 70, 70]
        cls = [0.1, 0.95]
        shape = (1, 2, 15)

        p_kpt = self._pred([_kpt_first_row([100, 50, 60, 20], 0.9, corners, cls)])
        p_cls = self._pred([_v5_row([100, 50, 60, 20], 0.9, corners, cls)])
        assert p_kpt.shape == p_cls.shape == shape

        _, s_kpt, c_kpt, pts_kpt = decode_predictions(
            p_kpt, resolve_output_spec(shape, num_classes=2, attr_order=ATTR_ORDER_KPT_FIRST)
        )
        _, s_cls, c_cls, pts_cls = decode_predictions(
            p_cls, resolve_output_spec(shape, num_classes=2, attr_order=ATTR_ORDER_CLS_FIRST)
        )

        # 各自读对了自己的那份数据：结果一致
        assert float(s_kpt[0]) == pytest.approx(float(s_cls[0]))
        assert int(c_kpt[0]) == int(c_cls[0]) == 1
        assert pts_kpt is not None and pts_cls is not None
        assert pts_kpt[0].tolist() == pts_cls[0].tolist()

        # 但把 kpt_first 的数据按 cls_first 读（即用户没设 attr_order）就会读错
        _, s_wrong, c_wrong, pts_wrong = decode_predictions(
            p_kpt, resolve_output_spec(shape, num_classes=2, attr_order=ATTR_ORDER_CLS_FIRST)
        )
        assert int(c_wrong[0]) != 1 or float(s_wrong[0]) != pytest.approx(float(s_kpt[0]))
        assert pts_wrong is not None and pts_wrong[0].tolist() != pts_kpt[0].tolist()

    def test_detection_only_returns_no_points(self):
        pred = self._pred([[10, 10, 8, 4, 0.1, 0.95]])
        _, _, _, points = decode_predictions(pred, resolve_output_spec(pred.shape, num_classes=2))
        assert points is None

    def test_channels_first_is_transposed_correctly(self):
        """(1, attr, N) 与 (1, N, attr) 表示的同一组目标必须解出同一结果。"""
        corners = [70, 30, 130, 30, 130, 70, 70, 70]
        row = [100, 50, 60, 20, 0.1, 0.95, *corners]  # v8-pose: box+cls+kpts
        n_first = self._pred([row, [0.0] * 14])
        c_first = np.transpose(n_first, (0, 2, 1))

        r1 = decode_predictions(n_first, resolve_output_spec(n_first.shape, num_classes=2))
        r2 = decode_predictions(c_first, resolve_output_spec(c_first.shape, num_classes=2))
        assert r1[0].tolist() == r2[0].tolist()
        assert r1[1].tolist() == r2[1].tolist()

    def test_attribute_dim_shorter_than_layout_is_rejected(self):
        spec = OutputSpec(False, True, 4, 2, 2, ATTR_ORDER_CLS_FIRST)
        with pytest.raises(ValueError):
            decode_predictions(np.zeros((1, 3, 5), np.float32), spec)


class TestNms:
    """非极大值抑制。"""

    def test_overlapping_boxes_keep_the_higher_score(self):
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], np.float32)
        scores = np.array([0.9, 0.8], np.float32)
        assert nms(boxes, scores, 0.45) == [0]

    def test_disjoint_boxes_both_survive(self):
        boxes = np.array([[0, 0, 10, 10], [500, 500, 510, 510]], np.float32)
        scores = np.array([0.9, 0.8], np.float32)
        assert sorted(nms(boxes, scores, 0.45)) == [0, 1]

    def test_class_aware_keeps_overlapping_different_classes(self):
        """单行牌与双层牌是两个类别，重叠时不该互相压掉。"""
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], np.float32)
        scores = np.array([0.9, 0.8], np.float32)
        classes = np.array([0, 1], np.int32)
        assert sorted(nms(boxes, scores, 0.45, classes)) == [0, 1]

    def test_result_is_sorted_by_descending_score(self):
        boxes = np.array([[0, 0, 10, 10], [500, 500, 510, 510]], np.float32)
        scores = np.array([0.3, 0.9], np.float32)
        keep = nms(boxes, scores, 0.45)
        assert keep == [1, 0]

    def test_empty_input(self):
        assert nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.5) == []


class TestLetterbox:
    """letterbox 与它的逆变换 —— 整条链路里最容易差半个像素的地方。"""

    def test_output_is_square_and_ratio_matches_short_side(self):
        img = np.zeros((480, 800, 3), np.uint8)
        padded, ratio, (pad_x, pad_y) = letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert ratio == pytest.approx(0.8)
        # 长边 800×0.8=640 铺满，短边 480×0.8=384 → 上下各补 128
        assert (pad_x, pad_y) == (0, 128)

    def test_padding_pairs_sum_to_the_gap(self):
        """填充之和 + 缩放后边长 必须正好等于目标边长，不允许再差更多。

        缺口是奇数时左右（上下）填充必然一多一少（如 128 + 129），
        所以这里允许 0~1 像素的不对称 —— 但**不能**出现 2 像素以上的偏差，
        那意味着两次独立取整把坐标整体推偏了。
        """
        for shape in ((480, 800), (801, 479), (101, 999)):
            h, w = shape
            img = np.zeros((h, w, 3), np.uint8)
            padded, ratio, (pad_x, pad_y) = letterbox(img, 640)
            assert padded.shape == (640, 640, 3)
            # 有任何一侧被填充时，角落的像素必须是灰边值而不是图像内容
            if pad_x or pad_y:
                assert padded[0, 0, 0] == 114
            assert 640 - (round(h * ratio) + 2 * pad_y) in (0, 1)
            assert 640 - (round(w * ratio) + 2 * pad_x) in (0, 1)

    def test_square_input_has_no_padding(self):
        img = np.zeros((640, 640, 3), np.uint8)
        _padded, ratio, pad = letterbox(img, 640)
        assert ratio == pytest.approx(1.0)
        assert pad == (0, 0)

    def test_coordinate_round_trip(self):
        """letterbox 坐标 → 原图坐标，与手算一致。

        桩模型写在 letterbox 坐标里的框是 (220,288)-(420,352)，
        逆变换应为 (275,200)-(525,280)。
        """
        img = np.zeros((480, 800, 3), np.uint8)
        _padded, ratio, (pad_x, pad_y) = letterbox(img, 640)
        x1, y1, x2, y2 = 220.0, 288.0, 420.0, 352.0
        back = [
            (x1 - pad_x) / ratio,
            (y1 - pad_y) / ratio,
            (x2 - pad_x) / ratio,
            (y2 - pad_y) / ratio,
        ]
        assert back == pytest.approx([275.0, 200.0, 525.0, 280.0])

    def test_degenerate_image_is_rejected(self):
        with pytest.raises(ValueError):
            letterbox(np.zeros((0, 10, 3), np.uint8), 640)


class TestCtcDecode:
    """CTC 贪心解码。车牌识别只有这一条后处理，错了就是整串错。"""

    CHARS: ClassVar[list[str]] = ["<blank>", "京", "A", "1", "B"]

    def _logits(self, indices: list[int]) -> np.ndarray:
        """造出 argmax 恰好是给定下标的 ``(T, C)`` logits。"""
        arr = np.full((len(indices), len(self.CHARS)), -10.0, np.float32)
        for t, i in enumerate(indices):
            arr[t, i] = 10.0
        return arr

    def test_collapses_repeats_and_drops_blank(self):
        """[京, blank, A, A, 1, blank] → 京A1（相邻重复合并）。"""
        text, confs, overall = ctc_decode(self._logits([1, 0, 2, 2, 3, 0]), self.CHARS)
        assert text == "京A1"
        assert len(confs) == 3
        assert 0.0 <= overall <= 1.0

    def test_blank_separated_repeats_are_kept(self):
        """[A, blank, A] → AA：先合再去 blank 会把它错并成一个。

        车牌里 ``京AA1234`` 这类真实重复字符很常见，合并错了直接读错号。
        """
        text, _confs, _overall = ctc_decode(self._logits([2, 0, 2]), self.CHARS)
        assert text == "AA"

    def test_all_blank_yields_empty_text(self):
        text, confs, overall = ctc_decode(self._logits([0, 0, 0]), self.CHARS)
        assert text == ""
        assert confs == []
        assert overall == 0.0

    def test_channels_first_output_is_transposed(self):
        """识别网络常导出 (C, T)，与 (T, C) 必须解出同一串。"""
        t_first = self._logits([1, 0, 2, 3])
        c_first = t_first.T
        assert c_first.shape != t_first.shape  # 非方阵，朝向真的可判
        assert ctc_decode(c_first, self.CHARS)[0] == ctc_decode(t_first, self.CHARS)[0]

    def test_confidence_is_mean_of_selected_timesteps(self):
        """整体置信度取被选中位置的均值，不是整张图的均值。"""
        _text, confs, overall = ctc_decode(self._logits([1, 0, 2]), self.CHARS)
        assert overall == pytest.approx(sum(confs) / len(confs))

    def test_orientation_uses_charset_length_not_size_comparison(self):
        """时间步比字符表**短**时也不能转错矩阵。

        早先按"谁小谁是字符表"猜朝向：字符表 70、时间步 5 的短车牌会让
        判别反过来，整串读成乱码。正确做法是用已知的字符表长度去认。
        """
        chars = ["<blank>", *[f"c{i}" for i in range(69)]]  # 70 个字符
        # 5 个时间步 → (T=5, C=70)，此时 T < C
        logits = np.full((5, 70), -10.0, np.float32)
        for t, idx in enumerate([0, 1, 0, 2, 0]):
            logits[t, idx] = 10.0
        assert ctc_decode(logits, chars)[0] == "c0c1"

    def test_charset_mismatch_is_rejected(self):
        """输出宽度与字符表都对不上时必须报错，而不是猜一个朝向接着算。

        这类不匹配几乎总是配置错误（字符表文件给错了），静默算下去
        会得到一串看着像车牌的乱码，比直接报错难查得多。
        """
        logits = np.full((6, 9), -10.0, np.float32)  # 9 既不是 5 也不是 6
        with pytest.raises(ValueError) as exc:
            ctc_decode(logits, self.CHARS)
        assert "字符表" in str(exc.value)


# ============================================================ 端到端（桩 ONNX）


@pytest.fixture(scope="module")
def stub_factory():
    """生成桩 ONNX 模型的工厂：``(目录, 布局, **kwargs) -> 模型路径``。

    放在模块作用域而不是类里：类作用域的 fixture 写成实例方法会触发
    pytest 的 ``PytestRemovedIn10Warning``（实例属性对测试方法不可见）。
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import make_stub_onnx

    def make(tmp_path: Path, layout: str, **kwargs: object) -> Path:
        out = tmp_path / layout / DETECT_MODEL_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        output = make_stub_onnx.build_output(layout, box=STUB_BOX, **kwargs)
        model = make_stub_onnx.build_model(output, input_size=STUB_INPUT)
        out.write_bytes(model.SerializeToString())
        return out

    return make


class TestEndToEndWithStubModel:
    """用桩 ONNX 跑通真实 onnxruntime 会话，验证坐标变换自洽。

    桩模型的输出是常量，所以解出来的框必须**精确等于手算值** ——
    任何 letterbox 逆变换、角点排序、类别列错位都会让它对不上。
    """

    @pytest.mark.parametrize(
        ("layout", "attr_order"),
        [
            ("v8-pose", ATTR_ORDER_CLS_FIRST),
            ("v5-pose", ATTR_ORDER_CLS_FIRST),
            ("kpt-first", ATTR_ORDER_KPT_FIRST),
            ("v8-det", ATTR_ORDER_CLS_FIRST),
            ("v5-det", ATTR_ORDER_CLS_FIRST),
        ],
    )
    def test_all_supported_layouts_decode_to_the_same_box(
        self, stub_factory, tmp_path, layout, attr_order
    ):
        """五种输出布局都必须解出同一个物理框 —— 这是"支持多种权重"的实证。"""
        model_path = stub_factory(tmp_path, layout)
        engine = YoloEngine(
            models_dir=model_path.parent,
            attr_order=attr_order,
            recognize_backend="none",
        )
        engine.load()

        img = np.full((480, 800, 3), 128, np.uint8)
        detections = engine._detect(img)

        assert len(detections) == 1, f"{layout} 应解出恰好 1 个候选"
        assert detections[0].box == pytest.approx((275.0, 200.0, 525.0, 280.0), abs=0.5)
        assert detections[0].layout is PlateLayout.SINGLE

    def test_duplicate_boxes_are_suppressed_by_nms(self, stub_factory, tmp_path):
        model_path = stub_factory(tmp_path, "v5-pose", duplicate=True)
        engine = YoloEngine(models_dir=model_path.parent, recognize_backend="none")
        engine.load()
        detections = engine._detect(np.full((480, 800, 3), 128, np.uint8))
        # 两条 IoU≈0.87 的框 → NMS 只留一条
        assert len(detections) == 1

    def test_double_class_maps_to_double_layout(self, stub_factory, tmp_path):
        model_path = stub_factory(tmp_path, "v5-pose", double=True)
        engine = YoloEngine(models_dir=model_path.parent, recognize_backend="none")
        engine.load()
        detections = engine._detect(np.full((480, 800, 3), 128, np.uint8))
        assert len(detections) == 1
        assert detections[0].layout is PlateLayout.DOUBLE

    def test_conf_threshold_filters_the_result(self, stub_factory, tmp_path):
        """阈值高于实际置信度时空结果 —— 证明阈值真的参与了决策。"""
        model_path = stub_factory(tmp_path, "v5-pose")  # 分数 0.855
        engine = YoloEngine(
            models_dir=model_path.parent,
            conf_threshold=0.99,
            recognize_backend="none",
        )
        engine.load()
        assert engine._detect(np.full((480, 800, 3), 128, np.uint8)) == []

    def test_geometry_filter_drops_absurd_aspect_ratio(self, stub_factory, tmp_path):
        """宽高比 10:1 的"框"不是车牌，必须在几何过滤里被丢掉。"""
        out = tmp_path / "odd" / DETECT_MODEL_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        import make_stub_onnx

        output = make_stub_onnx.build_output("v5-pose", box=(320.0, 320.0, 300.0, 20.0))
        out.write_bytes(
            make_stub_onnx.build_model(output, input_size=STUB_INPUT).SerializeToString()
        )

        engine = YoloEngine(models_dir=out.parent, recognize_backend="none")
        engine.load()
        assert engine._detect(np.full((480, 800, 3), 128, np.uint8)) == []

    def test_wrong_attr_order_produces_a_different_wrong_box(self, stub_factory, tmp_path):
        """把 kpt-first 权重按 cls-first 读，必须解出**不同**的框。

        这条断言存在的意义：证明 ``attr_order`` 不是装饰性参数。
        如果两种解读结果相同，说明切片逻辑没真的生效。
        """
        model_path = stub_factory(tmp_path, "kpt-first")
        good = YoloEngine(
            models_dir=model_path.parent,
            attr_order=ATTR_ORDER_KPT_FIRST,
            recognize_backend="none",
        )
        wrong = YoloEngine(
            models_dir=model_path.parent,
            attr_order=ATTR_ORDER_CLS_FIRST,
            recognize_backend="none",
        )
        good.load()
        wrong.load()

        img = np.full((480, 800, 3), 128, np.uint8)
        good_dets, wrong_dets = good._detect(img), wrong._detect(img)
        assert len(good_dets) == 1
        assert good_dets[0].box == pytest.approx((275.0, 200.0, 525.0, 280.0), abs=0.5)
        # 读错列顺序：框不再是那个常量框（或直接被过滤/阈值淘汰）
        if wrong_dets:
            assert wrong_dets[0].box != pytest.approx((275.0, 200.0, 525.0, 280.0), abs=0.5)

    def test_pure_detection_mode_returns_boxes_without_text(self, stub_factory, tmp_path):
        """没有识别后端时返回**带框无文本**的结果，而不是空列表。

        这不是失败而是明确降级：定位评测只需要框，静默返回空列表会让人
        以为模型坏了。
        """
        model_path = stub_factory(tmp_path, "v5-pose")
        engine = YoloEngine(models_dir=model_path.parent, recognize_backend="none")
        results = engine.predict(np.full((480, 800, 3), 128, np.uint8))

        assert len(results) == 1
        assert results[0].text == ""
        assert results[0].bbox is not None
        assert results[0].meta["recognize_backend"] == "none"
        assert results[0].engine == "yolo"
        # 框本身仍然可用：定位评测靠它
        assert results[0].roi is not None and results[0].roi.size > 0

    def test_warp_rectifies_a_rotated_plate(self, stub_factory, tmp_path):
        """侧倾的角点必须被矫正成正向矩形 —— 这正是引入检测引擎的理由。

        角点绕中心旋转 30° 后，矫正出来的图仍应是宽大于高的横矩形；
        若角点排序没做好（老版本 45° 以上会转成竖图），这里会立刻暴露。
        """
        import make_stub_onnx

        cx, cy, w, h = 320.0, 320.0, 200.0, 64.0
        theta = np.deg2rad(30.0)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        base = [
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
            (-w / 2, h / 2),
        ]
        rotated = []
        for dx, dy in base:
            rotated += [cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t]

        out = tmp_path / "rot" / DETECT_MODEL_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        output = make_stub_onnx.build_output("v5-pose", box=(cx, cy, w, h))
        # 直接替换角点列为旋转后的坐标
        meta = make_stub_onnx.LAYOUTS["v5-pose"]
        attr = int(meta["attr"](2))
        rows = output[0]  # (N, attr)
        rows[0, 5 : 5 + 8] = rotated
        assert rows.shape == (8, attr)
        out.write_bytes(
            make_stub_onnx.build_model(rows[None, ...], input_size=STUB_INPUT).SerializeToString()
        )

        engine = YoloEngine(models_dir=out.parent, recognize_backend="none")
        engine.load()
        detections = engine._detect(np.full((480, 800, 3), 128, np.uint8))
        assert len(detections) == 1
        roi = engine._warp(np.full((480, 800, 3), 128, np.uint8), detections[0].corners)
        assert roi is not None
        assert roi.shape[1] > roi.shape[0], "侧倾车牌被矫成了竖图（角点排序回归）"


class _FakeRecognizer:
    """假的"复用识别后端"：按调用方给的 bbox 决定**哪些候选有产出**。

    这个替身存在的唯一理由是复现一个真实行为：``deep`` 的
    ``recognize_regions`` 会把"一个字都没读出来"的候选**从返回列表里剔掉**，
    于是返回列表比入参短。用真引擎很难稳定构造这种场景（要挑一张恰好
    认不出的图），用替身则完全可控。
    """

    def __init__(self, text_for_bbox=None, *, drop_bboxes=()):
        self.text_for_bbox = text_for_bbox or {}
        self.drop_bboxes = set(drop_bboxes)
        self.seen: list[object] = []

    def recognize_regions(self, regions, *, layout_hint=None):
        results = []
        for i, (_roi, _color, bbox) in enumerate(regions):
            self.seen.append(bbox)
            if bbox in self.drop_bboxes:
                continue  # 模拟"多尺度全轮无产出 → continue"
            results.append(
                make_plate_result(
                    text=self.text_for_bbox.get(bbox, "京A12345"),
                    bbox=bbox,
                    conf=(0.9 - 0.1 * i),
                    engine="deep",
                )
            )
        return results


def make_plate_result(*, text: str, bbox, conf: float, engine: str):
    from lpr.types import PlateColor, PlateResult

    return PlateResult(
        text=text,
        color=PlateColor.BLUE,
        bbox=bbox,
        confidence=conf,
        engine=engine,
        meta={"ocr_backend": "rapidocr"},
    )


class TestReusedRecognizerAlignment:
    """复用 ``deep`` 识别管线时的**配对与保框**语义。

    这段逻辑出过一个真问题：早先按位置 ``zip(results, prepared)`` 配对，
    只要识别侧丢掉中间一个候选，后面全部错位 —— A 的号码会挂上 B 的框和
    检测分。而且掉掉的候选会**连框一起消失**，用户看到"未识别到车牌"，
    以为检测也失败了。
    """

    def _engine(self, stub_factory, tmp_path, recognizer):
        model_path = stub_factory(tmp_path, "v5-pose")
        engine = YoloEngine(models_dir=model_path.parent, recognize_backend="deep")
        engine.load()
        engine._recognizer_engine = recognizer  # 绕开懒加载的真 deep 引擎
        return engine

    def _two_detections(self, engine):
        """造两条检测：复制桩模型的框并整体右移，且**检测分刻意不同**。

        分数必须不同，否则"错位配对"就不可测 —— 桩模型两条框分数一样的话，
        把 A 的检测分贴到 B 身上也看不出来（这正是我第一版漏掉的）。
        """
        det = engine._detect(np.full((480, 800, 3), 128, np.uint8))[0]
        second = dataclasses.replace(
            det,
            box=(det.box[0] + 220.0, det.box[1], det.box[2] + 220.0, det.box[3]),
            corners=det.corners + np.array([220.0, 0.0], np.float32),
            score=det.score * 0.5,
        )
        return [det, second]

    def test_lost_candidate_keeps_its_box_with_empty_text(self, stub_factory, tmp_path):
        """识别侧丢掉的候选必须**保框留白**，而不是整条消失。"""
        engine = self._engine(stub_factory, tmp_path, _FakeRecognizer())
        dets = self._two_detections(engine)
        # 让第一条被识别侧丢掉
        engine._recognizer_engine = _FakeRecognizer(drop_bboxes={_to_xywh(dets[0])})

        results = engine._recognize_all(np.full((480, 800, 3), 128, np.uint8), dets)

        assert len(results) == 2, "丢掉的候选被静默吃掉了"
        empty = [r for r in results if r.text == ""]
        assert len(empty) == 1
        assert empty[0].meta.get("ocr_failed") is True
        assert empty[0].roi is not None and empty[0].roi.size > 0
        assert empty[0].bbox is not None
        # 保留下来的必须是**被丢掉的那一条**（分数是它的），不能张冠李戴
        assert tuple(empty[0].bbox) == _to_xywh(dets[0])
        assert empty[0].confidence == pytest.approx(float(dets[0].score), abs=1e-6)

    def test_results_are_matched_by_bbox_not_by_position(self, stub_factory, tmp_path):
        """第一条被丢掉时，第二条的号码**不能**错配到第一条的框上。"""
        engine = self._engine(stub_factory, tmp_path, _FakeRecognizer())
        dets = self._two_detections(engine)
        first_bbox = _to_xywh(dets[0])
        second_bbox = _to_xywh(dets[1])
        engine._recognizer_engine = _FakeRecognizer(
            text_for_bbox={second_bbox: "沪B67890"},
            drop_bboxes={first_bbox},
        )

        results = engine._recognize_all(np.full((480, 800, 3), 128, np.uint8), dets)

        recognized = [r for r in results if r.text]
        assert len(recognized) == 1
        # 号码属于第二条 → 它挂的必须是第二条的框
        assert tuple(recognized[0].bbox) == second_bbox
        # 检测分也必须来自自己那条（错位配对会把第一条的分贴上来）
        assert recognized[0].meta["det_score"] == pytest.approx(
            round(float(dets[1].score), 4), abs=1e-4
        )

    def test_all_recognized_leaves_nothing_to_backfill(self, stub_factory, tmp_path):
        """识别侧全都有产出时不得多补出空结果的框。"""
        engine = self._engine(stub_factory, tmp_path, _FakeRecognizer())
        dets = self._two_detections(engine)
        results = engine._recognize_all(np.full((480, 800, 3), 128, np.uint8), dets)
        assert len(results) == 2
        assert all(r.text for r in results)
        assert not any(r.meta.get("ocr_failed") for r in results)

    def test_reused_results_are_retagged_to_this_engine(self, stub_factory, tmp_path):
        """复用 deep 的结果要把 ``engine`` 改回 yolo，但保留"谁识别的"。"""
        engine = self._engine(stub_factory, tmp_path, _FakeRecognizer())
        dets = self._two_detections(engine)
        results = engine._recognize_all(np.full((480, 800, 3), 128, np.uint8), dets)
        for r in results:
            assert r.engine == "yolo"
        assert any(r.meta.get("recognizer") for r in results)
