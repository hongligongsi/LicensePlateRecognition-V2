"""统一高层 API —— 推荐应用层直接用这个模块。

把"引擎选择、级联、安全校验、脱敏、多目标聚合、结果导出"打包成一组
简单函数，避免调用方重复拼装这些逻辑。

典型用法::

    from lpr.api import Recognizer

    rec = Recognizer(engine="deep")           # 或 "auto" / "cv"
    result = rec.recognize("car.jpg")
    print(result.best.text)
    result.save_json("out.json")

    # 视频：自动做跨帧跟踪去重
    for plate in rec.recognize_video("parking.mp4"):
        print(plate.text)
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config
from .engines import describe_engines, get_engine
from .engines.base import BaseEngine, EngineUnavailable
from .plate_rules import describe as describe_plate
from .plate_rules import plate_score
from .tracker import MultiPlateTracker, merge_same_image
from .types import PlateResult
from .utils.image import draw_result, imread, imwrite
from .utils.logging_setup import get_logger
from .utils.security import (
    Limits,
    PrivacyPolicy,
    SecurityError,
    check_file,
    check_image,
    mask_plate,
    safe_output_path,
    scrub_value,
)

logger = get_logger("api")

#: 引擎级联顺序：需要"又快又准"时，先用 cv 快速筛，deep 负责精确识别
CASCADE_ORDER = ("cv", "deep")


@dataclass
class RecognizeOutcome:
    """一次识别的完整结果（含元信息）。"""

    source: str
    results: list[PlateResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    engine: str = ""
    #: 处理过程中发生的非致命问题（如某张图被安全策略拒绝）
    warnings: list[str] = field(default_factory=list)

    @property
    def best(self) -> PlateResult | None:
        """最可信的一个结果。"""
        if not self.results:
            return None
        return max(self.results, key=lambda r: (plate_score(r.text), r.confidence))

    @property
    def texts(self) -> list[str]:
        return [r.text for r in self.results]

    def to_dict(self, privacy: PrivacyPolicy | None = None) -> dict:
        """转成可序列化字典，按隐私策略决定是否打码。

        注意：不仅对 ``text`` 打码，还对 ``meta`` 等嵌套结构做**递归清洗** ——
        引擎会把候选明细写进 meta（如 ``variants=[["京AD77972", 1.2]]``），
        那里含明文号码，只处理 text 会绕过脱敏。
        """
        policy = privacy or PrivacyPolicy()
        payload = {
            "source": self.source,
            "engine": self.engine,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "count": len(self.results),
            "warnings": self.warnings,
            "plates": [
                {
                    **r.to_dict(),
                    "plate_type": describe_plate(r.text)["type"],
                }
                for r in self.results
            ],
        }
        # 整棵树过一遍脱敏，确保 meta / variants / source 里的号码都被覆盖
        return scrub_value(payload, policy)


@dataclass
class BatchOutcome:
    """批量识别的汇总。"""

    outcomes: list[RecognizeOutcome] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def hit(self) -> int:
        return sum(1 for o in self.outcomes if o.results)

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.started) * 1000

    @property
    def all_texts(self) -> list[str]:
        """所有识别到的号码（去重）。"""
        seen: dict[str, int] = {}
        for o in self.outcomes:
            for r in o.results:
                seen[r.text] = seen.get(r.text, 0) + 1
        return sorted(seen, key=lambda k: -seen[k])

    def save_json(self, path: str | Path, privacy: PrivacyPolicy | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        policy = privacy or PrivacyPolicy()
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": self.total,
            "hit": self.hit,
            "hit_rate": round(self.hit / self.total, 4) if self.total else 0.0,
            "elapsed_ms": round(self.elapsed_ms, 1),
            # 汇总字段同样要脱敏：否则 items 里打了码、这里又把明文列一遍，
            # 等于开了个后门。
            "unique_plates": [mask_plate(t, policy) for t in self.all_texts],
            "items": [o.to_dict(policy) for o in self.outcomes],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def save_csv(self, path: str | Path, privacy: PrivacyPolicy | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        policy = privacy or PrivacyPolicy()
        with path.open("w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                ["来源", "号码", "颜色", "类型", "置信度", "引擎", "bbox", "耗时ms"]
            )
            for o in self.outcomes:
                # 来源路径里可能嵌着车牌号（如 京AD77972_20260101.jpg），同样要打码
                source = mask_plate(o.source, policy) if policy.mask_output else o.source
                if not o.results:
                    writer.writerow([source, "", "", "", "", o.engine, "", f"{o.elapsed_ms:.0f}"])
                    continue
                for r in o.results:
                    writer.writerow(
                        [
                            source,
                            mask_plate(r.text, policy),
                            r.color.label,
                            describe_plate(r.text)["type"],
                            f"{r.confidence:.4f}",
                            r.engine,
                            str(r.bbox),
                            f"{o.elapsed_ms:.0f}",
                        ]
                    )
        return path


class Recognizer:
    """统一识别门面。

    封装了引擎选择、级联策略、安全校验、脱敏与多目标聚合。
    """

    def __init__(
        self,
        engine: str = "auto",
        config: Config | None = None,
        limits: Limits | None = None,
        privacy: PrivacyPolicy | None = None,
    ) -> None:
        """
        Args:
            engine: ``"auto"`` / ``"cv"`` / ``"deep"`` / ``"cascade"``。
                * ``auto``：优先 deep，不可用则退回 cv
                * ``cascade``：cv 定位筛候选 + deep 精确识别（最省时的组合）
            config: 全局配置。
            limits: 安全限制。
            privacy: 隐私策略。
        """
        self.config = config or Config.load()
        self.limits = limits or Limits()
        self.privacy = privacy or PrivacyPolicy()
        self.requested = engine
        self.engine_name = self._resolve_engine(engine)
        self._engine: BaseEngine | None = None

    # ------------------------------------------------------------------ 引擎解析

    @staticmethod
    def _resolve_engine(requested: str) -> str:
        """把请求的引擎名解析成实际可用的引擎名。"""
        infos = {i["name"]: i for i in describe_engines()}
        if requested == "auto":
            for name in ("deep", "cv"):
                if infos.get(name, {}).get("available"):
                    return name
            raise EngineUnavailable("没有可用的识别引擎，请安装 opencv-python 或 rapidocr-onnxruntime")
        if requested == "cascade":
            # 级联模式以 deep 为最终识别者，cv 只做定位
            if infos.get("deep", {}).get("available"):
                return "deep"
            if infos.get("cv", {}).get("available"):
                return "cv"
            raise EngineUnavailable("没有可用的识别引擎")
        return requested

    @property
    def engine(self) -> BaseEngine:
        """惰性加载引擎（首次访问时才加载模型）。"""
        if self._engine is None:
            self._engine = get_engine(self.engine_name, self.config.engine)
        return self._engine

    def warmup(self) -> None:
        """预热：提前加载模型，把首次识别的耗时挪到启动阶段。"""
        _ = self.engine

    # ------------------------------------------------------------------ 单图

    def recognize(self, source: str | Path | np.ndarray) -> RecognizeOutcome:
        """识别单张图片（路径或已加载的图像数组）。"""
        started = time.perf_counter()
        warnings: list[str] = []

        try:
            img = self._load(source)
        except (SecurityError, OSError) as exc:
            logger.warning("拒绝处理 %s: %s", source, exc)
            return RecognizeOutcome(
                source=str(source), elapsed_ms=(time.perf_counter() - started) * 1000,
                engine=self.engine_name, warnings=[str(exc)],
            )

        results = self._predict(img)
        results = merge_same_image(results)

        if not self.privacy.keep_roi:
            for r in results:
                r.roi = None

        return RecognizeOutcome(
            source=str(source),
            results=results,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            engine=self.engine_name,
            warnings=warnings,
        )

    def _load(self, source: str | Path | np.ndarray) -> np.ndarray:
        """按安全策略加载图像。"""
        if isinstance(source, np.ndarray):
            check_image(source, self.limits)
            return source

        path = check_file(source, self.limits)
        img = imread(path)
        check_image(img, self.limits)
        return img

    def _predict(self, img: np.ndarray) -> list[PlateResult]:
        """调用引擎识别，失败时给出可诊断的异常。"""
        try:
            return self.engine.predict(img) or []
        except EngineUnavailable:
            raise
        except Exception as exc:
            logger.exception("识别引擎异常")
            raise RuntimeError(f"识别失败: {exc}") from exc

    # ------------------------------------------------------------------ 批量

    def recognize_many(
        self, sources: list[str | Path], annotate_dir: str | Path | None = None
    ) -> BatchOutcome:
        """批量识别图片列表。

        Args:
            sources: 图片路径列表。
            annotate_dir: 若提供，则把标注图保存到该目录（防路径穿越）。
        """
        batch = BatchOutcome()
        if len(sources) > self.limits.max_batch_files:
            sources = list(sources)[: self.limits.max_batch_files]
            logger.warning("批量任务超过上限，已截断为 %d 个文件", self.limits.max_batch_files)

        for src in sources:
            outcome = self.recognize(src)
            batch.outcomes.append(outcome)

            if annotate_dir and outcome.results:
                try:
                    img = imread(check_file(src, self.limits))
                    annotated = draw_result(img, outcome.results)
                    target = safe_output_path(annotate_dir, Path(str(src)).name)
                    imwrite(target, annotated)
                except Exception as exc:
                    outcome.warnings.append(f"标注图保存失败: {exc}")
                    logger.debug("标注图保存失败 %s: %s", src, exc)

        return batch

    def recognize_dir(
        self,
        directory: str | Path,
        recursive: bool = False,
        annotate_dir: str | Path | None = None,
    ) -> BatchOutcome:
        """识别整个目录下的图片。"""
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"不是目录: {root}")

        pattern = "**/*" if recursive else "*"
        files = sorted(
            p
            for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in self.limits.allowed_extensions
        )
        logger.info("目录 %s 下找到 %d 张图片", root, len(files))
        return self.recognize_many(files, annotate_dir=annotate_dir)

    # ------------------------------------------------------------------ 视频

    def recognize_video(
        self,
        source: str | Path,
        stride: int = 10,
        output_video: str | Path | None = None,
        tracker: MultiPlateTracker | None = None,
    ) -> Iterator[PlateResult]:
        """流式识别视频，逐条 yield **新确认**的车牌。

        借助 ``MultiPlateTracker`` 做跨帧去重与投票，同一辆车只报一次，
        且用多帧结果纠正单帧的偶发错误。

        Args:
            source: 视频文件路径或摄像头编号字符串（如 ``"0"``）。
            stride: 每 N 帧识别一次（越大越快，越小越不容易漏）。
            output_video: 若提供，输出带标注的视频。
            tracker: 自定义跟踪器；默认创建一个合理的配置。
        """
        import cv2

        tracker = tracker or MultiPlateTracker()
        src: str | int = int(source) if str(source).isdigit() else str(source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if output_video:
            writer = cv2.VideoWriter(
                str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                if frame_idx % stride == 0:
                    try:
                        results = self._predict(frame)
                    except Exception as exc:
                        logger.warning("帧 %d 识别失败: %s", frame_idx, exc)
                        results = []

                    for plate in tracker.update(results):
                        yield plate

                    if writer is not None:
                        annotated = draw_result(frame, tracker.active())
                        writer.write(annotated)
                        frame = annotated

                if writer is not None and frame_idx % stride != 0:
                    writer.write(frame)
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        logger.info(
            "视频处理完成: %d 帧, 确认 %d 个车牌", frame_idx, len(tracker.all_reported())
        )

    # ------------------------------------------------------------------ 工具

    def engine_report(self) -> list[dict]:
        """引擎可用性报告，便于界面展示与问题排查。"""
        return describe_engines()

    def describe(self) -> dict:
        """当前识别器配置概览。"""
        return {
            "requested_engine": self.requested,
            "active_engine": self.engine_name,
            "limits": {
                "max_file_mb": self.limits.max_file_bytes / 1048576,
                "max_pixels": self.limits.max_pixels,
            },
            "privacy": {
                "mask_output": self.privacy.mask_output,
                "mask_logs": self.privacy.mask_logs,
                "keep_roi": self.privacy.keep_roi,
            },
            "engines": describe_engines(),
        }


__all__ = ["Recognizer", "RecognizeOutcome", "BatchOutcome", "CASCADE_ORDER"]
