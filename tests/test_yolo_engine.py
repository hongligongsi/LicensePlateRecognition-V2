"""检测识别引擎（``yolo``）的骨架测试。

**不依赖权重**：只验证骨架里已经实现的部分 —— 注册、可用性探测、
路径解析、错误提示的可诊断性。检测/识别后处理尚未实现（``NotImplementedError``
占位），待接入权重后在此补端到端用例。

设计意图（为什么这些断言值得存在）：
``yolo`` 引擎的权重不进仓库，用户装完 extras 后**大概率**看不到它出现在
下拉框里。此时唯一能帮到用户的就是「一条说明去哪拿权重的报错」。
v2.3.5 在 ``start.bat`` 上吃过反面教训（静默失败 → 用户以为代码坏了），
所以这里把「错误提示必须包含路径与安装命令」钉成测试。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lpr.engines import YoloEngine, available_engines, create_engine, describe_engines
from lpr.engines.yolo_engine import (
    DETECT_MODEL_NAME,
    RECOGNIZE_MODEL_NAME,
    Detection,
    locate_model_dir,
)

PROJECT = Path(__file__).resolve().parents[1]


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


class TestAvailabilityProbe:
    """可用性探测：onnxruntime 与权重**同时**具备才算可用。"""

    def test_unavailable_without_weights(self):
        """默认权重缺失时不可用（models/yolo 下没有 onnx）。

        提示必须可行动 —— 给出确切路径或安装命令，不能只说「不可用」。
        """
        ok, reason = YoloEngine.is_available()
        if ok:  # 已放入权重（例如 CI 里下载了）则不适用本断言
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
        if not engine.is_available()[0]:
            with pytest.raises(EngineUnavailable) as exc:
                engine.load()
            assert "plate_det.onnx" in str(exc.value)


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


class TestDetectionDataclass:
    """Detection 是检测后处理的输出契约，尺寸属性要稳。"""

    def test_width_height_from_box(self):
        import numpy as np

        from lpr.types import PlateLayout

        det = Detection(
            corners=np.zeros((4, 2), dtype=np.float32),
            box=(10.0, 20.0, 110.0, 50.0),
            layout=PlateLayout.SINGLE,
            score=0.9,
        )
        assert det.width == 100.0
        assert det.height == 30.0

    def test_corners_are_float32_quad(self):
        import numpy as np

        from lpr.types import PlateLayout

        det = Detection(
            corners=np.zeros((4, 2), dtype=np.float32),
            box=(0.0, 0.0, 10.0, 5.0),
            layout=PlateLayout.SINGLE,
            score=0.5,
        )
        assert det.corners.shape == (4, 2)
        assert det.corners.dtype == np.float32


class TestNotYetImplemented:
    """把「尚未实现」钉成测试 —— 接入权重时这些用例会失败，提醒补端到端测试。"""

    def test_detect_raises_until_weights_are_wired(self, tmp_path):
        """骨架阶段 _detect 应抛 NotImplementedError（有明确指引）。

        一旦有人接入权重实现了 _detect，本用例会失败 —— 这是**有意**的：
        那时必须补上真实权重的端到端精度/耗时对比，而不是悄悄删掉这个测试。
        """
        import numpy as np

        engine = YoloEngine(models_dir=tmp_path)
        # 绕过 ensure_ready 的加载检查，直接验证占位实现
        engine._ready = True
        with pytest.raises(NotImplementedError) as exc:
            engine._detect(np.zeros((100, 300, 3), dtype=np.uint8))
        assert "后处理" in str(exc.value) or "输出张量" in str(exc.value)
