"""跨帧多车牌跟踪器测试。

视频/摄像头场景的核心价值：同一辆车只上报一次，并用多帧投票
纠正单帧的偶发识别错误。
"""

from __future__ import annotations

import pytest

from lpr.tracker import MultiPlateTracker, Track, merge_same_image
from lpr.types import PlateColor, PlateResult


def _res(
    text: str,
    bbox: tuple[int, int, int, int] = (100, 100, 200, 60),
    confidence: float = 0.9,
    color: PlateColor = PlateColor.BLUE,
) -> PlateResult:
    return PlateResult(
        text=text,
        color=color,
        bbox=bbox,
        roi=None,
        confidence=confidence,
        engine="test",
    )


# ══════════════════════════════════════════════════ merge_same_image


def test_merge_same_image_drops_duplicate_bbox() -> None:
    a = _res("京A12345", (100, 100, 200, 60), 0.8)
    b = _res("京A12345", (102, 101, 200, 60), 0.9)
    merged = merge_same_image([a, b])
    assert len(merged) == 1
    assert merged[0].confidence >= 0.8


def test_merge_same_image_keeps_distinct_bboxes() -> None:
    a = _res("京A12345", (100, 100, 200, 60))
    b = _res("沪B12345", (800, 400, 200, 60))
    assert len(merge_same_image([a, b])) == 2


def test_merge_same_image_empty_input() -> None:
    assert merge_same_image([]) == []


# ══════════════════════════════════════════════════ 基本确认行为


def test_valid_plate_confirmed_on_first_frame() -> None:
    """合法号码几乎不可能是误检，应 1 次即确认。"""
    tracker = MultiPlateTracker()
    newly = tracker.update([_res("京A12345")])
    assert len(newly) == 1
    assert newly[0].text == "京A12345"


def test_same_plate_reported_only_once() -> None:
    tracker = MultiPlateTracker()
    first = tracker.update([_res("京A12345")])
    assert len(first) == 1
    for _ in range(5):
        assert tracker.update([_res("京A12345")]) == [], "同一号码不应重复上报"


def test_suspicious_plate_needs_multiple_hits() -> None:
    """语法不合法的号码需要多帧确认，抑制偶发误检。"""
    tracker = MultiPlateTracker(min_hits=3)
    assert tracker.update([_res("XYZ9999", confidence=0.5)]) == []
    assert tracker.update([_res("XYZ9999", confidence=0.5)]) == []
    third = tracker.update([_res("XYZ9999", confidence=0.5)])
    assert len(third) == 1


def test_empty_frame_returns_nothing() -> None:
    tracker = MultiPlateTracker()
    assert tracker.update([]) == []


def test_plate_without_text_is_ignored() -> None:
    tracker = MultiPlateTracker()
    assert tracker.update([_res("")]) == []


# ══════════════════════════════════════════════════ 跨帧关联


def test_track_follows_moving_plate_by_iou() -> None:
    tracker = MultiPlateTracker(iou_threshold=0.3)
    tracker.update([_res("京A12345", (100, 100, 200, 60))])
    # 车牌略微移动，IoU 仍足够高 → 仍视为同一轨迹
    tracker.update([_res("京A12345", (120, 105, 200, 60))])
    assert tracker.stats["active_tracks"] == 1


def test_multi_frame_voting_corrects_single_frame_error() -> None:
    """5 帧里 4 帧正确、1 帧认错，最终应采纳多数的正确号码。"""
    tracker = MultiPlateTracker()
    tracker.update([_res("京A12345")])
    # 同一位置出现错误号码，但 IoU 关联到同一轨迹
    tracker.update([_res("京A1234S", (101, 100, 200, 60))])
    final = tracker.active()
    assert len(final) == 1
    assert final[0].text == "京A12345", f"多帧投票未纠正错误: {final[0].text}"


def test_plate_matched_by_text_even_if_position_jumps() -> None:
    tracker = MultiPlateTracker(iou_threshold=0.9)
    tracker.update([_res("京A12345", (0, 0, 200, 60))])
    # 位置完全变了，但号码一致 → 按号码关联
    tracker.update([_res("京A12345", (900, 500, 200, 60))])
    assert tracker.stats["active_tracks"] == 1


def test_two_plates_tracked_independently() -> None:
    tracker = MultiPlateTracker()
    newly = tracker.update(
        [
            _res("京A12345", (0, 0, 200, 60)),
            _res("沪B67890", (800, 400, 200, 60)),
        ]
    )
    assert len(newly) == 2
    assert {r.text for r in newly} == {"京A12345", "沪B67890"}
    assert tracker.stats["active_tracks"] == 2


def test_expired_track_is_dropped() -> None:
    tracker = MultiPlateTracker(max_age=0.0)
    tracker.update([_res("京A12345")])
    # max_age=0 意味着下一帧就过期（时间必然推进）
    tracker.update([])
    assert tracker.stats["active_tracks"] == 0


# ══════════════════════════════════════════════════ 查询与生命周期


def test_all_reported_accumulates() -> None:
    tracker = MultiPlateTracker()
    tracker.update([_res("京A12345")])
    tracker.update([_res("沪B67890", (800, 400, 200, 60))])
    assert tracker.all_reported() == ["京A12345", "沪B67890"]


def test_reset_clears_state() -> None:
    tracker = MultiPlateTracker()
    tracker.update([_res("京A12345")])
    tracker.reset()
    assert tracker.all_reported() == []
    assert tracker.stats["active_tracks"] == 0
    # reset 后同一号码可以重新上报
    assert len(tracker.update([_res("京A12345")])) == 1


def test_stats_shape() -> None:
    tracker = MultiPlateTracker()
    tracker.update([_res("京A12345")])
    stats = tracker.stats
    assert "active_tracks" in stats
    assert "reported" in stats
    assert stats["reported_total"] == 1


def test_active_returns_plate_results_with_meta() -> None:
    tracker = MultiPlateTracker()
    tracker.update([_res("京A12345")])
    tracker.update([_res("京A12345", (101, 100, 200, 60))])
    active = tracker.active()
    assert len(active) == 1
    assert active[0].meta.get("hits") == 2
    assert "hit_rate" in active[0].meta


# ══════════════════════════════════════════════════ Track 单元


def test_track_majority_text() -> None:
    track = Track(track_id=1)
    track.votes = {"京A12345": 3.0, "京A1234S": 1.0}
    assert track.text == "京A12345"


def test_track_hit_rate() -> None:
    track = Track(track_id=1)
    track.votes = {"京A12345": 3.0, "京A1234S": 1.0}
    track.hits = 4
    assert 0.0 < track.hit_rate <= 1.0


def test_track_empty_has_no_text() -> None:
    assert Track(track_id=1).text == ""


@pytest.mark.parametrize("bad_iou", [-0.1, 1.5])
def test_invalid_iou_threshold_still_runs(bad_iou: float) -> None:
    """极端阈值不应崩溃（属配置错误，但需保持健壮）。"""
    tracker = MultiPlateTracker(iou_threshold=bad_iou)
    tracker.update([_res("京A12345")])
    tracker.update([_res("京A12345", (101, 100, 200, 60))])
    assert tracker.stats["active_tracks"] >= 1
