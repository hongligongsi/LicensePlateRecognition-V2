"""tools/eval_weak.py 的单元测试。

弱监督评测的价值在于"改定位参数有人拦"。工具本身若不可信，这道闸门就是
摆设，所以要测三件事：

* ``collect``：图片收集的排序 / 限量 / 扩展名过滤（0 = 全量）
* ``evaluate``：候选率 / 合法率 / 过检率的口径（用假引擎注入，不加载真模型）
* ``main`` 的 ``--override``：列表必须转成元组才能喂给 dataclass replace

真实引擎路径（RapidOCR）不在单元测试范围内 —— 那是弱监督评测自身的职责，
由 ``tools/eval_weak.py`` 在真实数据上人工执行。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import eval_weak  # noqa: E402
from lpr.config import LocateConfig  # noqa: E402
from lpr.types import PlateColor, PlateLayout, PlateResult  # noqa: E402


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def _make_image(path: Path, w: int = 32, h: int = 24) -> None:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """b.jpg / a.jpg / c.txt / sub/d.png —— 乱序、含非图片、含子目录。"""
    _make_image(tmp_path / "b.jpg")
    _make_image(tmp_path / "a.png")
    (tmp_path / "c.txt").write_text("not an image", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    _make_image(sub / "d.bmp")
    return tmp_path


def _result(text: str) -> PlateResult:
    return PlateResult(
        text=text,
        color=PlateColor.BLUE,
        layout=PlateLayout.SINGLE,
        bbox=(0, 0, 10, 4),
        roi=None,
        confidence=0.9,
        engine="fake",
        char_confidences=[0.9] * len(text),
        meta={},
    )


@pytest.fixture()
def patch_get_engine(monkeypatch: pytest.MonkeyPatch):
    """把 eval_weak 内部延迟导入的 get_engine 换成查表假引擎。"""

    def _install(script: dict[str, list[str]]):
        class _ScriptedEngine:
            name = "fake"

            def __init__(self):
                self.calls = 0

            def ensure_ready(self) -> None:
                pass

            def predict(self, image):
                self.calls += 1
                texts = script.get(self.calls, [])
                return [_result(t) for t in texts]

        from lpr import engines as engines_mod

        monkeypatch.setattr(engines_mod, "get_engine", lambda *a, **k: _ScriptedEngine())

    return _install


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
class TestCollect:
    def test_recursive_sorted_and_filters_extensions(self, image_dir: Path):
        files = eval_weak.collect(image_dir, 0)
        names = [f.name for f in files]
        # 只收图片扩展名，rglob 递归进子目录，按路径排序
        assert "c.txt" not in names
        assert set(names) == {"a.png", "b.jpg", "d.bmp"}

    def test_limit_takes_first_n(self, image_dir: Path):
        assert len(eval_weak.collect(image_dir, 2)) == 2

    def test_zero_limit_means_all(self, image_dir: Path):
        assert len(eval_weak.collect(image_dir, 0)) == 3

    def test_missing_dir_yields_nothing(self, tmp_path: Path):
        assert eval_weak.collect(tmp_path / "nope", 10) == []


# --------------------------------------------------------------------------- #
# evaluate 的指标口径
# --------------------------------------------------------------------------- #
class TestEvaluateMetrics:
    def _files(self, tmp_path: Path, n: int) -> list[Path]:
        files = []
        for i in range(n):
            p = tmp_path / f"img_{i}.jpg"
            _make_image(p)
            files.append(p)
        return files

    def test_counts_candidate_valid_and_overdetect(self, tmp_path: Path, patch_get_engine):
        # 6 张图：3 张合法、1 张非法文本、1 张双结果(过检)、1 张空
        patch_get_engine(
            {
                1: ["皖AM6160"],
                2: ["皖A2L112"],
                3: ["京A12345"],
                4: ["ABCDEFG"],  # 不合语法 → 不算合法
                5: ["皖A370J9", "皖B11111"],  # 两个结果 → 过检
                6: [],
            }
        )
        files = self._files(tmp_path, 6)
        rep = eval_weak.evaluate(files, LocateConfig(), "fake")
        assert rep["n_images"] == 6
        # 1-5 都有结果（6 为空）：4 号不合法但仍是候选，5 号双结果计过检
        assert rep["n_with_candidate"] == 5
        # 合法：1、2、3、5（5 号虽是双结果，但首个号码合法即计 1 次）
        assert rep["n_valid_text"] == 4
        assert rep["valid_rate"] == pytest.approx(4 / 6, abs=1e-3)
        assert rep["candidate_rate"] == pytest.approx(5 / 6, abs=1e-3)
        assert rep["over_detect_rate"] == pytest.approx(1 / 6, abs=1e-3)

    def test_engine_exception_does_not_crash_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from lpr import engines as engines_mod

        class _Boom:
            name = "boom"

            def ensure_ready(self) -> None:
                pass

            def predict(self, image):
                raise RuntimeError("模型炸了")

        monkeypatch.setattr(engines_mod, "get_engine", lambda *a, **k: _Boom())
        files = self._files(tmp_path, 2)
        rep = eval_weak.evaluate(files, LocateConfig(), "boom")
        assert rep["n_images"] == 2
        assert rep["n_with_candidate"] == 0
        assert rep["valid_rate"] == 0.0

    def test_unreadable_image_is_skipped(self, tmp_path: Path, patch_get_engine):
        patch_get_engine({1: ["皖AM6160"]})
        good = tmp_path / "good.jpg"
        _make_image(good)
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"\xff\xd8broken")  # 伪 JPEG，解码失败
        rep = eval_weak.evaluate([bad, good], LocateConfig(), "fake")
        assert rep["n_images"] == 1
        assert rep["valid_rate"] == 1.0

    def test_timing_fields_present(self, tmp_path: Path, patch_get_engine):
        patch_get_engine({1: ["皖AM6160"]})
        files = self._files(tmp_path, 1)
        rep = eval_weak.evaluate(files, LocateConfig(), "fake")
        assert rep["seconds"] >= 0
        assert rep["ms_per_image"] >= 0


# --------------------------------------------------------------------------- #
# select（skip/limit 切片口径）
# --------------------------------------------------------------------------- #
class TestSelect:
    def _files(self, n: int) -> list[Path]:
        return [Path(f"img_{i:03d}.jpg") for i in range(n)]

    def test_limit_takes_head(self):
        files = eval_weak.select(self._files(200), 0, 60)
        assert len(files) == 60
        assert files[0].name == "img_000.jpg"

    def test_zero_limit_means_all(self):
        assert len(eval_weak.select(self._files(50), 0, 0)) == 50

    def test_skip_only_takes_tail(self):
        files = eval_weak.select(self._files(50), 10, 0)
        assert len(files) == 40
        assert files[0].name == "img_010.jpg"

    def test_skip_plus_limit_is_absolute_window(self):
        """--skip 60 --limit 100 必须是第 61~160 张（共 100 张）。

        曾经的 bug：先 collect[:limit] 再切 skip，实际只评 40 张，
        调参段与验证段静默重叠 —— 这里钉死防复发。
        """
        files = eval_weak.select(self._files(654), 60, 100)
        assert len(files) == 100
        assert files[0].name == "img_060.jpg"
        assert files[-1].name == "img_159.jpg"

    def test_skip_beyond_end_yields_empty(self):
        assert eval_weak.select(self._files(10), 99, 5) == []

    def test_returns_copy_not_alias(self):
        src = self._files(5)
        out = eval_weak.select(src, 0, 0)
        assert out == src
        assert out is not src


# --------------------------------------------------------------------------- #
# main 的参数覆盖逻辑
# --------------------------------------------------------------------------- #
class TestMainOverride:
    def _run(self, monkeypatch, capsys, tmp_path, extra_args):
        images = tmp_path / "imgs"
        images.mkdir(exist_ok=True)
        _make_image(images / "a.jpg")

        calls: list[LocateConfig] = []

        def fake_evaluate(files, cfg, engine):
            calls.append(cfg)
            return {
                "n_images": 1,
                "n_with_candidate": 1,
                "n_valid_text": 1,
                "valid_rate": 0.5,
                "candidate_rate": 1.0,
                "over_detect_rate": 0.0,
                "seconds": 0.1,
                "ms_per_image": 100.0,
                "sample_texts": [],
            }

        monkeypatch.setattr(eval_weak, "evaluate", fake_evaluate)
        monkeypatch.setattr(
            sys,
            "argv",
            ["eval_weak.py", "--images", str(images), *extra_args],
        )
        rc = eval_weak.main()
        capsys.readouterr()
        return rc, calls

    def test_override_list_values_become_tuples(self, monkeypatch, capsys, tmp_path):
        rc, calls = self._run(
            monkeypatch,
            capsys,
            tmp_path,
            ["--override", '{"morph_kernel_ratio": [0.006, 0.03]}'],
        )
        assert rc == 0
        # 两次调用：先基线、后候选（候选值必须与默认值不同，否则断言无意义）
        assert len(calls) == 2
        assert calls[1].morph_kernel_ratio == (0.006, 0.03)
        assert calls[0].morph_kernel_ratio != (0.006, 0.03)

    def test_scalar_override_applied(self, monkeypatch, capsys, tmp_path):
        rc, calls = self._run(
            monkeypatch,
            capsys,
            tmp_path,
            ["--override", '{"min_area_ratio": 0.0008}'],
        )
        assert rc == 0
        assert calls[1].min_area_ratio == pytest.approx(0.0008)
        # 基线保持默认值不被污染
        assert calls[0].min_area_ratio == LocateConfig().min_area_ratio

    def test_no_override_runs_baseline_only(self, monkeypatch, capsys, tmp_path):
        rc, calls = self._run(monkeypatch, capsys, tmp_path, [])
        assert rc == 0
        assert len(calls) == 1

    def test_json_output_written(self, monkeypatch, capsys, tmp_path):
        out = tmp_path / "rep.json"
        rc, _ = self._run(
            monkeypatch,
            capsys,
            tmp_path,
            ["--json", str(out), "--override", '{"color_ratio": 0.35}'],
        )
        assert rc == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["delta_valid_rate"] == 0.0
        assert "candidate" in data
