"""高层 API（Recognizer / RecognizeOutcome / BatchOutcome）测试。

重点覆盖**脱敏契约**：任何落盘格式（dict / JSON / CSV）都不得泄露明文号码。
"""

from __future__ import annotations

import json
from pathlib import Path

from lpr.api import BatchOutcome, RecognizeOutcome
from lpr.types import PlateColor, PlateResult
from lpr.utils.security import PrivacyPolicy


def _plate(text: str, conf: float = 0.9, variants=None) -> PlateResult:
    meta = {}
    if variants is not None:
        meta["variants"] = variants
    return PlateResult(
        text=text,
        color=PlateColor.BLUE,
        bbox=(10, 10, 200, 60),
        roi=None,
        confidence=conf,
        engine="test",
        meta=meta,
    )


def _outcome(*texts: str, variants=None) -> RecognizeOutcome:
    return RecognizeOutcome(
        source="samples/car.jpg",
        results=[_plate(t, 0.9 - i * 0.1, variants) for i, t in enumerate(texts)],
        elapsed_ms=123.4,
        engine="deep",
    )


# ══════════════════════════════════════════════ RecognizeOutcome


def test_best_picks_highest_scoring_plate() -> None:
    out = RecognizeOutcome(
        source="x.jpg",
        results=[_plate("ABC", 0.99), _plate("京A12345", 0.5)],
    )
    assert out.best is not None
    assert out.best.text == "京A12345", "合法号码应优先于高置信度的废串"


def test_best_of_empty_is_none() -> None:
    assert RecognizeOutcome(source="x.jpg").best is None


def test_texts_lists_all() -> None:
    out = _outcome("京A12345", "沪B67890")
    assert out.texts == ["京A12345", "沪B67890"]


def test_to_dict_plain_when_no_privacy() -> None:
    data = _outcome("京A12345").to_dict()
    assert data["plates"][0]["text"] == "京A12345"
    assert data["engine"] == "deep"
    assert data["elapsed_ms"] == 123.4


def test_to_dict_includes_plate_type() -> None:
    data = _outcome("京AD77972").to_dict()
    assert data["plates"][0]["plate_type"] == "新能源"


def test_to_dict_masks_text_when_requested() -> None:
    data = _outcome("京AD77972").to_dict(PrivacyPolicy(mask_output=True))
    assert data["plates"][0]["text"] != "京AD77972"
    assert "D779" not in json.dumps(data, ensure_ascii=False)


def test_to_dict_masks_nested_variants() -> None:
    """关键回归：meta.variants 里的明文号码必须一并打掉。"""
    out = _outcome("京AD77972", variants=[["京AD77972", 1.2], ["京AD77973", 0.3]])
    data = out.to_dict(PrivacyPolicy(mask_output=True))
    blob = json.dumps(data, ensure_ascii=False)
    assert "京AD77972" not in blob, f"variants 泄露明文: {blob}"
    assert "京AD77973" not in blob, f"variants 泄露明文: {blob}"


def test_to_dict_masks_source_path_plate() -> None:
    out = RecognizeOutcome(
        source="dumps/京AD77972_20260101.jpg",
        results=[_plate("京AD77972")],
    )
    data = out.to_dict(PrivacyPolicy(mask_output=True))
    assert "京AD77972" not in json.dumps(data, ensure_ascii=False)


def test_to_dict_is_json_serializable() -> None:
    data = _outcome("京A12345").to_dict()
    json.dumps(data, ensure_ascii=False)  # 不应抛异常


# ══════════════════════════════════════════════ BatchOutcome


def _batch(*outcomes: RecognizeOutcome) -> BatchOutcome:
    return BatchOutcome(outcomes=list(outcomes), started=0.0)


def test_batch_total_and_hit() -> None:
    batch = _batch(_outcome("京A12345"), RecognizeOutcome(source="b.jpg"))
    assert batch.total == 2
    assert batch.hit == 1


def test_batch_all_texts_dedupes_and_ranks() -> None:
    batch = _batch(
        _outcome("京A12345"),
        _outcome("京A12345"),
        _outcome("沪B67890"),
    )
    texts = batch.all_texts
    assert set(texts) == {"京A12345", "沪B67890"}
    assert texts[0] == "京A12345", "出现次数多的应排在前面"


def test_batch_empty() -> None:
    batch = _batch()
    assert batch.total == 0
    assert batch.hit == 0
    assert batch.all_texts == []


def test_save_json_roundtrip(tmp_path: Path) -> None:
    batch = _batch(_outcome("京A12345"))
    path = batch.save_json(tmp_path / "r.json")
    assert path.exists()
    data = json.loads(path.read_text("utf-8"))
    assert data["total"] == 1
    assert data["hit"] == 1
    assert data["unique_plates"] == ["京A12345"]
    assert data["items"][0]["plates"][0]["text"] == "京A12345"


def test_save_json_applies_masking(tmp_path: Path) -> None:
    batch = _batch(_outcome("京AD77972", variants=[["京AD77972", 1.0]]))
    path = batch.save_json(tmp_path / "m.json", PrivacyPolicy(mask_output=True))
    blob = path.read_text("utf-8")
    assert "京AD77972" not in blob, f"JSON 落盘泄露明文: {blob}"


def test_save_csv_creates_file(tmp_path: Path) -> None:
    batch = _batch(_outcome("京A12345"))
    path = batch.save_csv(tmp_path / "r.csv")
    assert path.exists()
    content = path.read_text("utf-8-sig")
    assert "京A12345" in content


def test_save_csv_applies_masking(tmp_path: Path) -> None:
    batch = _batch(_outcome("京AD77972"))
    path = batch.save_csv(tmp_path / "m.csv", PrivacyPolicy(mask_output=True))
    content = path.read_text("utf-8-sig")
    assert "京AD77972" not in content, f"CSV 落盘泄露明文: {content}"


def test_save_json_creates_parent_dirs(tmp_path: Path) -> None:
    batch = _batch(_outcome("京A12345"))
    path = batch.save_json(tmp_path / "deep" / "nested" / "r.json")
    assert path.exists()
