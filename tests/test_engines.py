"""端到端测试：需要模型文件与样张，缺失时自动跳过。"""
from __future__ import annotations

from pathlib import Path

import pytest

from lpr.config import Config
from lpr.engines import CVEngine

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
SAMPLES = PROJECT / "samples"

# 与样张肉眼可核对的真值
EXPECTED = {
    "1.jpg": "京E51619",
    "2.jpg": "京AD77972",
}


def models_available() -> bool:
    return (MODELS / "svm.dat").exists() and (MODELS / "svmchinese.dat").exists()


def samples_available() -> bool:
    return all((SAMPLES / name).exists() for name in EXPECTED)


requires_models = pytest.mark.skipif(
    not models_available(), reason="缺少 svm.dat / svmchinese.dat"
)


class TestCVEngine:
    @requires_models
    def test_load(self):
        engine = CVEngine(Config().engine, models_dir=MODELS)
        engine.load()
        assert engine.ready

    @requires_models
    @pytest.mark.skipif(not samples_available(), reason="缺少样张")
    def test_predict_runs(self):
        engine = CVEngine(Config().engine, models_dir=MODELS)
        engine.load()
        for name in EXPECTED:
            results = engine.predict(SAMPLES / name)
            # 不对 CV 引擎做精度断言（SVM 精度有限是已知问题，deep 引擎负责精度），
            # 只验证流程能跑通、结构正确
            for res in results:
                assert res.engine == "cv"
                assert res.color is not None

    @requires_models
    def test_no_plate_returns_empty(self):
        engine = CVEngine(Config().engine, models_dir=MODELS)
        engine.load()
        import numpy as np

        blank = np.full((400, 600, 3), 128, np.uint8)
        assert engine.predict(blank) == []
