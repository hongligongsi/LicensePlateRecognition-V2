"""多目标聚合与跨帧跟踪。

两个使用场景：

1. **单图多车牌**：一张照片里有好几辆车，需要去重、排序、避免重复输出。
2. **视频流跨帧**：同一辆车在连续帧里被反复识别，需要"同一辆车只报一次"，
   并且用多帧结果投票来提高准确率（单帧可能错一个字，多帧投票能纠正）。

这是"更全面"能力的核心：不再把每次识别当作孤立的单张图片处理。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .plate_rules import plate_score
from .types import PlateResult
from .utils.logging_setup import get_logger

logger = get_logger("tracker")


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """两个轴对齐矩形的交并比。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def merge_same_image(results: list[PlateResult], iou_threshold: float = 0.4) -> list[PlateResult]:
    """合并同一张图内的重复结果。

    重复的判定（满足任一即视为同一个车牌）：
      * 号码完全相同
      * 检测框重叠超过阈值（同一车牌被不同候选重复识别）
      * 号码相似且框有重叠（差一个字符的同一车牌）

    保留策略：号码更合法、置信度更高者胜出。
    """
    kept: list[PlateResult] = []

    for res in sorted(results, key=lambda r: (-plate_score(r.text), -r.confidence)):
        duplicate = False
        for exist in kept:
            if exist.text == res.text:
                duplicate = True
                break
            if exist.bbox and res.bbox:
                overlap = _iou(exist.bbox, res.bbox)
                if overlap > iou_threshold:
                    # 框重叠且号码相近 → 是同一个车牌的不同识别结果
                    if _similar(exist.text, res.text):
                        duplicate = True
                        break
                    # 框高度重叠但号码差异大，通常是误检，丢弃分低的
                    if overlap > 0.85:
                        duplicate = True
                        break
        if not duplicate:
            kept.append(res)

    return kept


def _similar(a: str, b: str) -> bool:
    """两个号码是否只差 1~2 个字符（视为同一车牌的识别差异）。"""
    if not a or not b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) != len(b):
        return False
    diff = sum(1 for x, y in zip(a, b, strict=False) if x != y)
    return diff <= 2


@dataclass
class Track:
    """一辆车的跟踪记录。"""

    track_id: int
    #: 号码 → 累计票数
    votes: dict[str, float] = field(default_factory=dict)
    bbox: tuple[int, int, int, int] | None = None
    color: Any = None  # PlateColor（延迟赋值，保持宽松以兼容部分识别结果缺色）
    #: 首帧时间戳
    first_seen: float = field(default_factory=time.time)
    #: 最后出现时间
    last_seen: float = field(default_factory=time.time)
    #: 出现帧数
    hits: int = 0
    #: 最佳 ROI 快照
    best_roi: Any = None  # np.ndarray | None
    best_conf: float = 0.0

    @property
    def text(self) -> str:
        """当前得票最高的号码。"""
        if not self.votes:
            return ""
        return max(self.votes, key=lambda k: self.votes[k])

    @property
    def hit_rate(self) -> float:
        """最高号号码占总票数的比例，作为稳定性指标。"""
        total = sum(self.votes.values()) or 1.0
        return self.votes.get(self.text, 0.0) / total

    def to_result(self) -> PlateResult:
        """把跟踪结果转成标准识别结果。"""
        text = self.text
        return PlateResult(
            text=text,
            color=self.color,
            bbox=self.bbox,
            roi=self.best_roi,
            confidence=float(min(1.0, self.best_conf * (0.5 + 0.5 * self.hit_rate))),
            engine="track",
            char_confidences=[self.best_conf] * len(text),
            meta={
                "track_id": self.track_id,
                "hits": self.hits,
                "hit_rate": round(self.hit_rate, 3),
                "variants": sorted(self.votes.items(), key=lambda kv: -kv[1])[:3],
                "duration_s": round(self.last_seen - self.first_seen, 2),
            },
        )


class MultiPlateTracker:
    """跨帧多车牌跟踪器。

    用于视频/摄像头流：把连续帧的识别结果按位置关联成"轨迹"，
    同一辆车只输出一次，并用多帧投票纠正单帧的偶发错误。

    用法::

        tracker = MultiPlateTracker(iou_threshold=0.3, max_age=2.0)
        for frame in video:
            results = engine.predict(frame)
            confirmed = tracker.update(results)
            for r in confirmed:      # 只会返回"已稳定"的新车牌
                print("新识别到:", r.text)
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: float = 2.0,
        min_hits: int = 2,
        confirm_score: float = 0.8,
    ) -> None:
        """
        Args:
            iou_threshold: 关联同一轨迹所需的框重叠度。
            max_age: 轨迹多少秒没更新就丢弃。
            min_hits: 至少被识别到几次才认为该车牌"确认"（过滤偶发误检）。
            confirm_score: 语法合法性达到该值时允许只出现 1 次就确认
                （合法号码几乎不可能是误检，无需多次确认）。
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.confirm_score = confirm_score

        self._tracks: list[Track] = []
        self._next_id = 1
        #: 已经对外"确认"过的号码，避免重复上报
        self._reported: set[str] = set()

    # ------------------------------------------------------------------ 主循环

    def update(self, results: list[PlateResult]) -> list[PlateResult]:
        """送入一帧的识别结果，返回**新确认**的车牌列表。"""
        now = time.time()
        results = merge_same_image(results)

        matched_tracks: set[int] = set()
        matched_results: set[int] = set()

        # 1) 先用 IoU 关联已有轨迹
        for ti, track in enumerate(self._tracks):
            if not track.bbox:
                continue
            best_i, best_overlap = -1, self.iou_threshold
            for ri, res in enumerate(results):
                if ri in matched_results or not res.bbox:
                    continue
                overlap = _iou(track.bbox, res.bbox)
                if overlap > best_overlap:
                    best_i, best_overlap = ri, overlap

            if best_i >= 0:
                self._absorb(track, results[best_i], now)
                matched_tracks.add(ti)
                matched_results.add(best_i)

        # 2) 号码匹配：位置变了但号码一致，也算同一辆车
        for ti, track in enumerate(self._tracks):
            if ti in matched_tracks:
                continue
            for ri, res in enumerate(results):
                if ri in matched_results:
                    continue
                if res.text and res.text == track.text:
                    self._absorb(track, res, now)
                    matched_tracks.add(ti)
                    matched_results.add(ri)
                    break

        # 3) 剩下的识别结果建立新轨迹
        for ri, res in enumerate(results):
            if ri in matched_results or not res.text:
                continue
            track = Track(track_id=self._next_id)
            self._next_id += 1
            self._absorb(track, res, now)
            self._tracks.append(track)
            matched_tracks.add(len(self._tracks) - 1)

        # 4) 淘汰过期轨迹
        self._tracks = [t for t in self._tracks if now - t.last_seen <= self.max_age]

        # 5) 收集"新确认"的车牌
        newly: list[PlateResult] = []
        for track in self._tracks:
            text = track.text
            if not text or text in self._reported:
                continue

            # 合法号码一次即可确认；不合法/可疑的需要多次确认
            score = plate_score(text)
            needed = 1 if score >= self.confirm_score else self.min_hits
            if track.hits >= needed:
                self._reported.add(text)
                newly.append(track.to_result())
                logger.info(
                    "确认新车牌 %s（出现 %d 次, 一致率 %.2f, 合法性 %.2f）",
                    text,
                    track.hits,
                    track.hit_rate,
                    score,
                )

        return newly

    def _absorb(self, track: Track, res: PlateResult, now: float) -> None:
        """把一次观测并入轨迹。"""
        if res.text:
            weight = res.confidence * (0.5 + plate_score(res.text))
            track.votes[res.text] = track.votes.get(res.text, 0.0) + weight
        track.bbox = res.bbox or track.bbox
        track.color = res.color or track.color
        track.last_seen = now
        track.hits += 1
        if res.confidence > track.best_conf and res.roi is not None:
            track.best_roi = res.roi
            track.best_conf = res.confidence

    # ------------------------------------------------------------------ 查询

    def active(self) -> list[PlateResult]:
        """当前所有活跃轨迹（不只新确认的）。"""
        return [t.to_result() for t in self._tracks if t.text]

    def all_reported(self) -> list[str]:
        """本次会话已确认过的全部号码。"""
        return sorted(self._reported)

    def reset(self) -> None:
        """清空跟踪状态（切换视频源时调用）。"""
        self._tracks.clear()
        self._reported.clear()
        self._next_id = 1

    @property
    def stats(self) -> dict:
        return {
            "active_tracks": len(self._tracks),
            "reported_total": len(self._reported),
            "reported": sorted(self._reported),
        }


__all__ = ["MultiPlateTracker", "Track", "merge_same_image"]
