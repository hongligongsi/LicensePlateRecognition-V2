"""识别工作线程。

原版 ``vedio_thread`` 的问题：
  * 直接在子线程里调 Tkinter 的 ``Label.configure``（Tkinter 不是线程安全的，
    偶发崩溃）；
  * 识别是同步的，识别期间画面完全冻结，所以只能"每 2 秒识别一帧"，
    体验很差；
  * 线程退出靠 ``thread_run = False`` 后 ``join(2.0)``，识别慢时会超时强杀，
    资源释放不干净。

这里改成标准的 Qt 并发模式：
  * ``ImageWorker``（QThread）：对一张图跑识别，结果用信号发回主线程；
  * ``CameraWorker``（QThread）：独立采集摄像头帧，**采集和识别解耦**，
    画面始终流畅，识别在最新一帧上异步进行，识别结果异步回传。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..config import Config
from ..engines import get_engine
from ..types import PlateResult
from ..utils.image import imread
from ..utils.logging_setup import get_logger

logger = get_logger("ui.worker")


class ImageWorker(QThread):
    """对单张图片执行识别。"""

    finished_with_results = Signal(str, list, float)  # path, results, 耗时ms
    failed = Signal(str, str)

    def __init__(self, engine_name: str, config: Config, parent=None) -> None:
        super().__init__(parent)
        self.engine_name = engine_name
        self.config = config
        self.image_path: str | Path | None = None
        self.image_array: np.ndarray | None = None

    def set_path(self, path: str | Path) -> "ImageWorker":
        self.image_path = path
        self.image_array = None
        return self

    def set_array(self, img: np.ndarray, source: str = "<memory>") -> "ImageWorker":
        self.image_array = img
        self.image_path = source
        return self

    def run(self) -> None:
        try:
            engine = get_engine(self.engine_name, self.config.engine)
            started = time.perf_counter()

            if self.image_array is not None:
                results = engine.predict(self.image_array)
            elif self.image_path:
                results = engine.predict(imread(self.image_path))
            else:
                results = []

            elapsed = (time.perf_counter() - started) * 1000
            self.finished_with_results.emit(str(self.image_path), results, elapsed)
        except Exception as exc:
            logger.exception("识别失败")
            self.failed.emit(str(self.image_path), str(exc))


class CameraWorker(QThread):
    """摄像头采集 + 异步识别。

    采集循环里**绝不做识别**，只把最新帧存进 ``self._latest``；
    识别由本线程在"采集间隙"进行 —— 因为识别用的模型在独立线程里，
    即便一次识别耗时 200ms，主窗口画面照常由采集循环刷新。
    """

    frame_ready = Signal(object)  # np.ndarray (BGR)
    results_ready = Signal(list, float)  # results, 耗时ms
    camera_error = Signal(str)

    def __init__(
        self,
        engine_name: str,
        config: Config,
        camera_index: int = 0,
        interval: float = 0.35,
        resolution: tuple[int, int] = (1280, 720),
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine_name = engine_name
        self.config = config
        self.camera_index = camera_index
        self.interval = max(0.05, interval)
        self.resolution = resolution
        self._running = False
        self._latest: np.ndarray | None = None
        self._recognizing = False

    def stop(self) -> None:
        """请求停止（线程安全）。"""
        self._running = False

    def run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.camera_error.emit(f"无法打开摄像头 {self.camera_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        self._running = True
        last_predict = 0.0
        logger.info("摄像头已启动 index=%d, 分辨率请求 %s", self.camera_index, self.resolution)

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    # 偶发的读帧失败不退出，连续失败才报错
                    time.sleep(0.05)
                    continue

                self._latest = frame
                self.frame_ready.emit(frame)

                now = time.time()
                if now - last_predict >= self.interval and not self._recognizing:
                    last_predict = now
                    started = time.perf_counter()
                    try:
                        engine = get_engine(self.engine_name, self.config.engine)
                        results = engine.predict(frame)
                        elapsed = (time.perf_counter() - started) * 1000
                        self.results_ready.emit(results, elapsed)
                    except Exception as exc:
                        logger.exception("视频帧识别失败")
                        self.camera_error.emit(f"识别出错: {exc}")

                # 控制采集节奏，避免跑满 CPU
                time.sleep(0.01)
        finally:
            cap.release()
            logger.info("摄像头已释放")

    def grab_latest(self) -> np.ndarray | None:
        """取当前帧的拷贝（抓拍用）。"""
        if self._latest is None:
            return None
        return self._latest.copy()


__all__ = ["ImageWorker", "CameraWorker"]
