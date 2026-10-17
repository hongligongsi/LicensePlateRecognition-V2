"""v2.3.1 修复项回归测试。

覆盖四个真 Bug 的修复，以及两项配套改动（``mode`` 语义、引擎单例配置感知）：

1. ``Recognizer`` 此前静默忽略 ``config.security``（``LPR_MASK_OUTPUT`` 环境变量、
   ``config.json`` 的 security 段对 Python API 无效）；
2. GUI 此前完全没有接入安全层（不校验输入、CSV 明文导出、日志脱敏未安装）；
3. ``CameraWorker`` 此前把 ``predict`` 写在采集循环里同步执行，识别会拖慢画面；
4. 模型哈希白名单此前**无处配置**，且"未校验"被显示成"✓ 通过"。

这些测试的用意是：任何一项被改回原样，都必须有测试失败。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lpr.api import Recognizer
from lpr.config import Config, EngineConfig, LocateConfig, SecurityConfig
from lpr.engines import registry as engine_registry
from lpr.engines.base import BaseEngine
from lpr.plate_rules import is_valid
from lpr.types import PlateColor, PlateResult
from lpr.utils.security import (
    Limits,
    MaskingFilter,
    ModelStatus,
    SecurityError,
    check_model,
    verify_model,
)

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "lpr"


# ══════════════════════════════════════════════════════════════════
# 假引擎夹具
# ══════════════════════════════════════════════════════════════════


class _Recorder:
    """记录调用情况的可控假引擎。"""

    calls: ClassVar[list[np.ndarray]] = []
    delay: float = 0.0
    text: str = "京A12345"


def _make_fake(name: str, text: str):
    class FakeEngine(BaseEngine):
        display_name = f"假引擎 {name}"
        description = "仅用于测试"

        def load(self) -> None:
            self._ready = True

        def predict(self, image):
            _Recorder.calls.append(image)
            if _Recorder.delay:
                time.sleep(_Recorder.delay)
            if not text:
                return []
            return [
                PlateResult(
                    text=text,
                    color=PlateColor.BLUE,
                    bbox=(2, 2, 40, 12),
                    confidence=0.93,
                    engine=name,
                )
            ]

    FakeEngine.name = name
    return FakeEngine


@pytest.fixture
def registered(monkeypatch):
    """注册/卸载一批假引擎，避免污染全局注册表。"""
    _Recorder.calls = []
    _Recorder.delay = 0.0
    engines = {
        "fakea": _make_fake("fakea", ""),  # 永远不命中
        "fakeb": _make_fake("fakeb", "京A12345"),  # 稳定命中
        "fakec": _make_fake("fakec", "京A12345"),
    }
    for key, cls in engines.items():
        engine_registry._REGISTRY[key] = cls
    yield engines
    for key in engines:
        engine_registry._REGISTRY.pop(key, None)
        engine_registry._INSTANCES.pop(key, None)
        engine_registry._INSTANCE_KEYS.pop(key, None)
    _Recorder.calls = []


@pytest.fixture
def clean_masking():
    """测试前后还原 lpr 各级 logger 的过滤器，避免测试互相影响。"""
    names = ("lpr", "lpr.engines", "lpr.engines.cv", "lpr.engines.deep", "lpr.ui")
    loggers = [logging.getLogger(n) for n in names]
    before = {lg.name: list(lg.filters) for lg in loggers}
    yield logging.getLogger("lpr")
    for lg in loggers:
        lg.filters[:] = before[lg.name]


# ══════════════════════════════════════════════════════════════════
# 一、Recognizer 必须回落到 config.security（Bug 1）
# ══════════════════════════════════════════════════════════════════


def test_recognizer_inherits_config_security():
    """核心回归：不显式传 limits/privacy 时，必须用 config.security。"""
    cfg = Config()
    cfg.security.mask_output = True
    cfg.security.max_file_mb = 1.0
    cfg.security.keep_roi = False

    rec = Recognizer(engine="cv", config=cfg)

    assert rec.privacy.mask_output is True, "config.security 的打码策略被忽略了"
    assert rec.limits.max_file_bytes == int(1.0 * 1024 * 1024), "config 的大小上限被忽略"
    assert rec.privacy.keep_roi is False


def test_recognizer_explicit_limits_override_config():
    """显式传参优先级高于 config，行为不能被改坏。"""
    cfg = Config()
    cfg.security.max_file_mb = 1.0
    explicit = Limits(max_file_bytes=7 * 1024 * 1024)

    rec = Recognizer(engine="cv", config=cfg, limits=explicit)

    assert rec.limits is explicit
    assert rec.limits.max_file_bytes == 7 * 1024 * 1024


def test_env_mask_output_reaches_python_api(monkeypatch):
    """``LPR_MASK_OUTPUT=1`` 必须对 Python API 也生效（此前只对 CLI 生效）。"""
    monkeypatch.setenv("LPR_MASK_OUTPUT", "1")
    cfg = Config.load()
    assert cfg.security.mask_output is True

    rec = Recognizer(engine="cv", config=cfg)
    assert rec.privacy.mask_output is True


def test_recognizer_installs_log_masking(clean_masking):
    """config.security.mask_logs 默认 True，应装上日志脱敏过滤器。"""
    cfg = Config()
    cfg.security.mask_logs = True

    Recognizer(engine="cv", config=cfg)

    assert any(isinstance(f, MaskingFilter) for f in clean_masking.filters)


def test_recognizer_log_masking_can_be_disabled(clean_masking):
    """mask_logs=False 时不应安装过滤器（该开关必须有实际作用）。"""
    clean_masking.filters[:] = [
        f for f in clean_masking.filters if not isinstance(f, MaskingFilter)
    ]
    cfg = Config()
    cfg.security.mask_logs = False

    Recognizer(engine="cv", config=cfg)

    assert not any(isinstance(f, MaskingFilter) for f in clean_masking.filters)


# ══════════════════════════════════════════════════════════════════
# 二、EngineConfig.mode 必须有真实语义（不再是死配置）
# ══════════════════════════════════════════════════════════════════


def test_mode_single_has_no_fallback(registered):
    cfg = Config()
    cfg.engine.mode = "single"
    rec = Recognizer(engine="fakea", config=cfg)
    assert rec.fallback_name is None


def test_mode_cascade_prepares_fallback(registered, monkeypatch):
    monkeypatch.setattr("lpr.api.CASCADE_ORDER", ("fakeb",))
    cfg = Config()
    cfg.engine.mode = "cascade"
    rec = Recognizer(engine="fakea", config=cfg)
    assert rec.fallback_name == "fakeb"


def test_cascade_skips_primary_engine_itself(registered, monkeypatch):
    """兜底引擎不能是主引擎自己。"""
    monkeypatch.setattr("lpr.api.CASCADE_ORDER", ("fakea", "fakeb"))
    cfg = Config()
    cfg.engine.mode = "cascade"
    rec = Recognizer(engine="fakea", config=cfg)
    assert rec.fallback_name == "fakeb"


def test_cascade_fallback_actually_used(registered, monkeypatch):
    """主引擎返回空结果时，mode=cascade 必须真的走兜底引擎。"""
    monkeypatch.setattr("lpr.api.CASCADE_ORDER", ("fakeb",))
    cfg = Config()
    cfg.engine.mode = "cascade"
    rec = Recognizer(engine="fakea", config=cfg)

    out = rec.recognize(np.zeros((40, 120, 3), np.uint8))

    assert [r.text for r in out.results] == ["京A12345"]
    assert out.results[0].engine == "fakeb"


def test_cascade_not_used_when_primary_hits(registered, monkeypatch):
    """主引擎命中时不应再跑兜底引擎（避免双倍开销）。"""
    monkeypatch.setattr("lpr.api.CASCADE_ORDER", ("fakea",))
    cfg = Config()
    cfg.engine.mode = "cascade"
    rec = Recognizer(engine="fakeb", config=cfg)

    out = rec.recognize(np.zeros((40, 120, 3), np.uint8))

    assert out.results[0].engine == "fakeb"
    assert len(_Recorder.calls) == 1, "主引擎命中却仍跑了兜底引擎"


def test_mode_documented_as_orthogonal_to_default():
    """default 与 mode 是两个正交维度，默认值不应自相矛盾。

    这里**不要**再抄一份引擎白名单：抄一份就会和 ``config.ENGINE_CHOICES`` 走散
    —— 本用例就因为 v2.8.1 把默认值改成策略名 ``auto`` 而失效过一次。
    """
    from lpr.config import ENGINE_CHOICES

    cfg = EngineConfig()
    assert cfg.default in ENGINE_CHOICES
    assert cfg.mode in ("single", "cascade")
    # 正交性：mode 换了不影响 default 的取值合法性，反之亦然
    assert EngineConfig(default=cfg.default, mode="cascade").default == cfg.default
    assert EngineConfig(default="deep", mode=cfg.mode).mode == cfg.mode


# ══════════════════════════════════════════════════════════════════
# 三、引擎单例必须感知配置变化
# ══════════════════════════════════════════════════════════════════


def test_get_engine_rebuilds_on_config_change(registered):
    cfg1 = EngineConfig(locate=LocateConfig(max_side=800))
    cfg2 = EngineConfig(locate=LocateConfig(max_side=1600))

    e1 = engine_registry.get_engine("fakec", cfg1, preload=False)
    e2 = engine_registry.get_engine("fakec", cfg2, preload=False)

    assert e1 is not e2, "配置变了却复用了旧单例，新配置被静默忽略"


def test_get_engine_reuses_instance_for_same_config(registered):
    cfg = EngineConfig()
    e1 = engine_registry.get_engine("fakec", cfg, preload=False)
    e2 = engine_registry.get_engine("fakec", cfg, preload=False)
    assert e1 is e2


def test_clear_instances_resets_config_keys(registered):
    engine_registry.get_engine("fakec", EngineConfig(), preload=False)
    engine_registry.clear_instances()
    assert "fakec" not in engine_registry._INSTANCES
    assert "fakec" not in engine_registry._INSTANCE_KEYS


def test_inference_lock_is_exported():
    """推理串行锁必须是公开 API（UI 线程与批量线程都要用它）。"""
    from lpr.engines import INFERENCE_LOCK

    with INFERENCE_LOCK:
        pass


# ══════════════════════════════════════════════════════════════════
# 四、模型完整性校验的三态（Bug 4）
# ══════════════════════════════════════════════════════════════════


def test_check_model_skipped_is_not_ok(tmp_path):
    """关键回归：未配置白名单必须是 skipped，而不是"通过"。"""
    p = tmp_path / "m.dat"
    p.write_bytes(b"data")

    result = check_model(p, None)

    assert result.status is ModelStatus.SKIPPED
    # 跳过校验不能算作"已验证"
    assert result.status.verified is False
    # 跳过校验不能显示成 ✓
    assert result.status.symbol == "—"


def test_check_model_mismatch(tmp_path):
    p = tmp_path / "m.dat"
    p.write_bytes(b"tampered")

    result = check_model(p, "a" * 64)

    assert result.status is ModelStatus.MISMATCH
    assert result.ok is False
    assert result.status.symbol == "✗"


def test_check_model_missing(tmp_path):
    result = check_model(tmp_path / "nope.dat", None)
    assert result.status is ModelStatus.MISSING
    assert result.ok is False


def test_check_model_ok(tmp_path):
    import hashlib

    p = tmp_path / "m.dat"
    p.write_bytes(b"good")
    result = check_model(p, hashlib.sha256(b"good").hexdigest())

    assert result.status is ModelStatus.OK
    assert result.status.verified is True
    assert result.status.symbol == "✓"


def test_verify_model_backward_compatible(tmp_path):
    """旧的二元 API 语义保持不变（跳过校验仍返回 True）。"""
    p = tmp_path / "m.dat"
    p.write_bytes(b"data")
    ok, msg = verify_model(p, None)
    assert ok is True
    assert "跳过" in msg


def test_default_config_ships_real_model_hashes():
    """默认配置必须自带随机型分发的哈希，否则校验永远只是"跳过"。"""
    cfg = SecurityConfig()
    assert "svm.dat" in cfg.model_hashes
    assert "svmchinese.dat" in cfg.model_hashes
    assert all(len(v) == 64 for v in cfg.model_hashes.values())


def test_shipped_models_match_shipped_hashes():
    """仓库里的模型文件必须与内置哈希一致（否则用户一装就报篡改）。"""
    from lpr.config import MODELS_DIR

    cfg = SecurityConfig()
    checked = 0
    for name, expected in cfg.model_hashes.items():
        path = MODELS_DIR / name
        if not path.exists():
            continue
        result = check_model(path, expected)
        assert result.status is ModelStatus.OK, f"{name}: {result.message}"
        checked += 1
    assert checked > 0, "没找到任何模型文件，测试本身失效"


def test_model_hashes_survive_config_roundtrip(tmp_path):
    cfg = Config()
    path = cfg.save(tmp_path / "config.json")
    loaded = Config.load(path)

    assert loaded.security.model_hashes == cfg.security.model_hashes


def test_config_json_can_override_model_hashes(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"security": {"model_hashes": {"custom.dat": "abc"}}}', encoding="utf-8")
    cfg = Config.load(p)
    assert cfg.security.model_hashes == {"custom.dat": "abc"}


def test_security_config_to_context_carries_hashes(registered):
    cfg = SecurityConfig()
    ctx = cfg.to_context()
    assert ctx.model_hashes == cfg.model_hashes
    assert ctx.install_log_masking is cfg.mask_logs


# ══════════════════════════════════════════════════════════════════
# 五、GUI 安全接线与采集/识别解耦（Bug 2 / Bug 3）
# ══════════════════════════════════════════════════════════════════


def _read_source(rel: str) -> str:
    return (SRC_ROOT / rel).read_text(encoding="utf-8")


def test_image_worker_uses_config_limits():
    """ImageWorker 的 limits 必须来自 config.security，而不是库默认值。"""
    pytest.importorskip("PySide6")
    from lpr.ui.worker import ImageWorker

    cfg = Config()
    cfg.security.max_file_mb = 2.0
    w = ImageWorker("cv", cfg)
    assert w.limits.max_file_bytes == int(2.0 * 1024 * 1024)


def test_image_worker_rejects_oversized_image():
    """超限像素数的图像必须被安全层拒绝（此前 GUI 无任何校验）。"""
    pytest.importorskip("PySide6")
    from lpr.ui.worker import ImageWorker

    cfg = Config()
    cfg.security.max_pixels = 100
    w = ImageWorker("cv", cfg)
    w.set_array(np.zeros((50, 50, 3), np.uint8))

    with pytest.raises(SecurityError):
        w._load()


def test_image_worker_loads_valid_image():
    pytest.importorskip("PySide6")
    from lpr.ui.worker import ImageWorker

    cfg = Config()
    w = ImageWorker("cv", cfg)
    img = np.zeros((30, 90, 3), np.uint8)
    w.set_array(img)
    assert w._load() is img


def test_camera_worker_does_not_predict_in_capture_loop():
    """核心回归：采集线程里不能出现模型推理调用。"""
    src = _read_source("ui/worker.py")
    capture_body = src.split("class CameraWorker", 1)[1]

    assert "engine.predict(" not in capture_body, (
        "CameraWorker 又变成同步识别了：识别会阻塞采集，画面会掉帧"
    )
    # 那段「只赋值、从不读取」的死逻辑不能回来（文档里提到这个字段名不算）
    assert "self._recognizing" not in src


def test_camera_worker_runs_recognition_on_separate_thread():
    src = _read_source("ui/worker.py")
    assert "class _RecognizeThread(QThread)" in src
    assert "self._recog.start()" in src


def test_recognize_thread_only_recognizes_new_frames(registered):
    """同一帧（序号未变）不得重复识别。"""
    pytest.importorskip("PySide6")
    from lpr.ui.worker import _RecognizeThread

    _Recorder.delay = 0.0
    frame = np.zeros((32, 96, 3), np.uint8)
    t = _RecognizeThread("fakeb", Config(), interval=0.0, limits=Limits())
    t.bind_source(lambda: (frame, 7))  # 序号恒定 → 永远同一帧
    t.start()
    time.sleep(0.4)
    t.stop()
    t.wait(2000)

    assert len(_Recorder.calls) == 1, f"同一帧被识别了 {len(_Recorder.calls)} 次"


def test_recognize_thread_keeps_up_with_advancing_frames(registered):
    """序号在推进时应持续识别，说明识别线程没有拖住采集。"""
    pytest.importorskip("PySide6")
    from lpr.ui.worker import _RecognizeThread

    _Recorder.delay = 0.01
    state = {"seq": 0}

    def source():
        state["seq"] += 1
        return np.zeros((32, 96, 3), np.uint8), state["seq"]

    t = _RecognizeThread("fakeb", Config(), interval=0.0, limits=Limits())
    t.bind_source(source)
    t.start()
    deadline = time.time() + 5
    while len(_Recorder.calls) < 5 and time.time() < deadline:
        time.sleep(0.02)
    t.stop()
    t.wait(2000)

    assert len(_Recorder.calls) >= 5


def test_camera_worker_stop_stops_recognition_thread():
    pytest.importorskip("PySide6")
    from lpr.ui.worker import CameraWorker

    cam = CameraWorker("cv", Config())
    assert cam._recog is not None
    cam.stop()
    assert cam._running is False
    assert cam._recog._running is False


def test_window_applies_security_context():
    """GUI 必须走与 CLI 同一份 config.security（防回退护栏）。"""
    src = _read_source("ui/window.py")
    assert "config.security.to_context()" in src
    assert "check_file(" in src
    assert "check_image(" in src


def test_window_masks_csv_export():
    """CSV 导出必须脱敏，不能再明文写车牌号。"""
    src = _read_source("ui/window.py")
    export = src.split("def on_export_history", 1)[1]
    assert "mask_plate(" in export
    assert "self.privacy" in export


def test_window_caps_batch_size():
    src = _read_source("ui/window.py")
    assert "max_batch_files" in src


# ══════════════════════════════════════════════════════════════════
# 六、级联兜底不能破坏既有正字法
# ══════════════════════════════════════════════════════════════════


def test_fallback_result_is_still_validated_by_rules(registered, monkeypatch):
    """兜底引擎的结果同样要过语法校验，不能因为"兜底"就放松。"""
    monkeypatch.setattr("lpr.api.CASCADE_ORDER", ("fakeb",))
    cfg = Config()
    cfg.engine.mode = "cascade"
    rec = Recognizer(engine="fakea", config=cfg)

    out = rec.recognize(np.zeros((40, 120, 3), np.uint8))

    assert out.best is not None
    assert is_valid(out.best.text)
