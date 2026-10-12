"""批量识别多进程加速的测试。

覆盖点：
* ``resolve_workers`` 的边界（0=自动、上限、不超过文件数）
* ``make_chunks`` 的完整性与不重不漏
* ``Config.to_payload / from_payload`` 往返
* ``PlateResult.from_dict`` / ``RecognizeOutcome.from_dict`` 的反向还原
* 多进程结果**顺序与串行一致**、条数与号码一致
* 单条失败被隔离成 warning，不拖垮整批
* ``workers > 1`` 时 ROI 不回传（已文档化的取舍）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lpr.api import Recognizer
from lpr.batch import (
    MAX_WORKERS,
    MIN_FILES_PER_WORKER,
    make_chunks,
    parallel_is_worthwhile,
    resolve_workers,
    run_parallel,
)
from lpr.config import Config
from lpr.types import PlateColor, PlateLayout, PlateResult
from lpr.utils.image import imwrite


@pytest.fixture
def force_parallel(monkeypatch):
    """把"值得并行"的门槛降到 1，让端到端测试真的走多进程路径。"""
    monkeypatch.setattr("lpr.batch.MIN_FILES_PER_WORKER", 1)


# ══════════════════════════════════════════════════════════════════
# 一、进程数与分片
# ══════════════════════════════════════════════════════════════════


def test_resolve_workers_zero_means_auto(monkeypatch):
    monkeypatch.setattr("lpr.batch._cpu_budget", lambda: 8)
    assert resolve_workers(0, 100) == 7  # 可调度 CPU 数 - 1


def test_resolve_workers_zero_single_cpu(monkeypatch):
    monkeypatch.setattr("lpr.batch._cpu_budget", lambda: 1)
    assert resolve_workers(0, 100) == 1


def test_resolve_workers_single_item():
    assert resolve_workers(0, 1) == 1
    assert resolve_workers(8, 1) == 1


def test_resolve_workers_never_exceeds_file_count():
    assert resolve_workers(64, 3) == 3


def test_resolve_workers_capped_by_max():
    assert resolve_workers(9999, 1000) == MAX_WORKERS


def test_resolve_workers_negative_is_auto(monkeypatch):
    monkeypatch.setattr("lpr.batch._cpu_budget", lambda: 4)
    assert resolve_workers(-5, 100) == 3


def test_parallel_not_worthwhile_for_tiny_batches():
    """小批量并行会更慢，必须主动退回串行（实测依据见 batch.py 注释）。"""
    assert parallel_is_worthwhile(MIN_FILES_PER_WORKER - 1, 4) is False
    assert parallel_is_worthwhile(MIN_FILES_PER_WORKER * 4, 4) is True


def test_parallel_not_worthwhile_for_single_worker():
    assert parallel_is_worthwhile(10_000, 1) is False


def test_recognize_many_falls_back_to_serial_for_tiny_batch(tmp_path, caplog):
    """文件太少时即使指定 workers>1 也应走串行（不启动进程池）。"""
    import logging

    paths = []
    for i in range(3):
        p = tmp_path / f"tiny_{i}.jpg"
        imwrite(p, np.zeros((20, 60, 3), np.uint8))
        paths.append(p)

    with caplog.at_level(logging.INFO, logger="lpr.api"):
        batch = Recognizer(engine="cv").recognize_many(paths, workers=4)

    assert batch.total == 3
    assert any("走串行" in r.message for r in caplog.records)


def test_cpu_budget_never_zero():
    from lpr.batch import _cpu_budget

    assert _cpu_budget() >= 1


def test_make_chunks_covers_all_items_exactly_once():
    sources = [f"img_{i}.jpg" for i in range(23)]
    chunks = make_chunks(sources, 4)

    flat = [p for _, part in chunks for p in part]
    assert flat == sources, "分片必须不重不漏且保持顺序"
    assert len(chunks) > 1
    assert [idx for idx, _ in chunks] == list(range(len(chunks)))


def test_make_chunks_uses_more_chunks_than_workers():
    """分片应多于进程数，才能做负载均衡。"""
    sources = [f"{i}.jpg" for i in range(40)]
    assert len(make_chunks(sources, 4)) > 4


def test_make_chunks_single_worker():
    assert make_chunks(["a.jpg", "b.jpg"], 1) == [(0, ["a.jpg", "b.jpg"])]


def test_make_chunks_handles_paths():
    chunks = make_chunks([Path("a.jpg"), Path("b.jpg")], 2)
    flat = [p for _, part in chunks for p in part]
    assert flat == ["a.jpg", "b.jpg"], "必须转成字符串才能跨进程传递"


# ══════════════════════════════════════════════════════════════════
# 二、配置与结果的序列化往返
# ══════════════════════════════════════════════════════════════════


def test_config_payload_roundtrip():
    cfg = Config()
    cfg.engine.locate.max_side = 1234
    cfg.engine.mode = "cascade"
    cfg.security.mask_output = True
    cfg.ui.camera_interval = 0.25

    back = Config.from_payload(cfg.to_payload())

    assert back.engine.locate.max_side == 1234
    assert back.engine.mode == "cascade"
    assert back.security.mask_output is True
    assert back.ui.camera_interval == 0.25
    assert back.engine.locate.morph_kernel_ratio == cfg.engine.locate.morph_kernel_ratio


def test_config_payload_is_json_safe():
    import json

    payload = Config().to_payload()
    # 不能有 tuple / dataclass 残留，否则 pickle 与 JSON 都不稳
    json.dumps(payload)


def test_config_from_payload_tolerates_empty():
    assert Config.from_payload({}).engine.default == "cascade"


def test_plate_result_from_dict_roundtrip():
    original = PlateResult(
        text="京A12345",
        color=PlateColor.BLUE,
        layout=PlateLayout.SINGLE,
        bbox=(3, 4, 90, 30),
        confidence=0.8123,
        engine="deep",
        char_confidences=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        meta={"path": "guided"},
    )

    back = PlateResult.from_dict(original.to_dict())

    assert back.text == original.text
    assert back.color is PlateColor.BLUE
    assert back.layout is PlateLayout.SINGLE
    assert back.bbox == (3, 4, 90, 30)
    assert back.confidence == pytest.approx(0.8123, abs=1e-4)
    assert back.engine == "deep"
    assert len(back.char_confidences) == 7
    assert back.meta == {"path": "guided"}
    assert back.roi is None, "ROI 不在 to_dict 里，还原后必须是 None"


def test_plate_result_from_dict_handles_missing_fields():
    back = PlateResult.from_dict({})
    assert back.text == ""
    assert back.color is PlateColor.UNKNOWN
    assert back.layout is PlateLayout.UNKNOWN
    assert back.bbox is None
    assert back.char_confidences == []


def test_plate_result_from_dict_tolerates_bad_enum():
    back = PlateResult.from_dict({"text": "X", "color": "not-a-color", "layout": "???"})
    assert back.color is PlateColor.UNKNOWN
    assert back.layout is PlateLayout.UNKNOWN


def test_recognize_outcome_from_dict_roundtrip():
    from lpr.api import RecognizeOutcome

    original = RecognizeOutcome(
        source="a.jpg",
        results=[PlateResult(text="京A12345", engine="deep", bbox=(1, 1, 10, 5))],
        elapsed_ms=12.5,
        engine="deep",
        warnings=["w1"],
    )

    back = RecognizeOutcome.from_dict(original.to_dict())

    assert back.source == "a.jpg"
    assert back.engine == "deep"
    assert back.elapsed_ms == pytest.approx(12.5, abs=1e-3)
    assert back.warnings == ["w1"]
    assert back.texts == ["京A12345"]


def test_recognize_outcome_from_dict_handles_empty():
    from lpr.api import RecognizeOutcome

    back = RecognizeOutcome.from_dict({})
    assert back.results == []
    assert back.best is None


# ══════════════════════════════════════════════════════════════════
# 三、多进程端到端
# ══════════════════════════════════════════════════════════════════


def _write_plate_like(path: Path, seed: int = 0) -> None:
    """写一张"看着像车牌"的合成图（蓝底白字块），便于 cv 引擎出候选。"""
    h, w = 60, 200
    img = np.full((h, w, 3), (200, 120, 20), np.uint8)  # 蓝底
    band = img[8 : h - 8, 12 : w - 12]
    band[:] = (235, 235, 235)
    # 画几条竖条模拟字符
    for i in range(7):
        x0 = 16 + i * 24 + seed % 3
        img[16 : h - 16, x0 : x0 + 12] = (30, 30, 30)
    imwrite(path, img)


@pytest.fixture
def image_set(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"plate_{i:02d}.jpg"
        _write_plate_like(p, seed=i)
        paths.append(p)
    return paths


@pytest.mark.slow
def test_parallel_matches_serial_count_and_order(image_set, force_parallel):
    serial = Recognizer(engine="cv").recognize_many(image_set, workers=1)
    parallel = Recognizer(engine="cv").recognize_many(image_set, workers=2)

    assert parallel.total == serial.total == len(image_set)
    assert [o.source for o in parallel.outcomes] == [o.source for o in serial.outcomes], (
        "多进程结果的顺序必须与输入顺序一致"
    )
    assert parallel.all_texts == serial.all_texts
    assert parallel.hit == serial.hit


@pytest.mark.slow
def test_parallel_drops_roi(image_set, force_parallel):
    """已文档化的取舍：多进程不回传 ROI。"""
    parallel = Recognizer(engine="cv").recognize_many(image_set, workers=2)
    assert all(r.roi is None for o in parallel.outcomes for r in o.results)


@pytest.mark.slow
def test_parallel_isolates_broken_files(tmp_path, image_set, force_parallel):
    """单个坏文件只记 warning，不能让整批崩掉。"""
    missing = tmp_path / "does_not_exist.jpg"
    sources = [str(image_set[0]), str(missing), str(image_set[1])]

    batch = Recognizer(engine="cv").recognize_many(sources, workers=2)

    assert batch.total == 3
    broken = [o for o in batch.outcomes if o.source.endswith("does_not_exist.jpg")]
    assert len(broken) == 1
    assert broken[0].warnings, "坏文件必须留下可诊断的 warning"
    # 好文件不应该受到牵连（不产生额外 warning）
    good = [o for o in batch.outcomes if not o.source.endswith("does_not_exist.jpg")]
    assert len(good) == 2
    assert all(not o.warnings for o in good)


@pytest.mark.slow
def test_parallel_single_worker_path_still_serial(image_set):
    """workers=1 必须走原串行路径（不启动进程池）。"""
    batch = Recognizer(engine="cv").recognize_many(image_set, workers=1)
    assert batch.total == len(image_set)


def test_run_parallel_empty_input():
    assert run_parallel([], "cv", Config().to_payload(), 4) == []


@pytest.mark.slow
def test_recognize_dir_passes_workers(tmp_path, image_set, force_parallel):
    rec = Recognizer(engine="cv")
    batch = rec.recognize_dir(tmp_path, workers=2)
    assert batch.total == len(image_set)


def test_cli_exposes_workers_option():
    from lpr.cli import _make_parser

    parser = _make_parser()
    args = parser.parse_args(["predict", "a.jpg", "--workers", "4"])
    assert args.workers == 4
    assert parser.parse_args(["predict", "a.jpg"]).workers == 1


def test_recognize_many_documents_roi_tradeoff():
    """取舍必须写在 docstring 里，避免使用者踩坑。"""
    doc = Recognizer.recognize_many.__doc__ or ""
    assert "roi" in doc.lower()
    assert "workers" in doc
