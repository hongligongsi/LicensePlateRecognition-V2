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
  * ``CameraWorker``（QThread）：只负责**采集**，把最新帧记在 ``_latest``；
  * ``_RecognizeThread``（QThread）：只负责**识别**，始终取"最新一帧"推理。

.. note::
   v2.3.1 修复：此前 ``CameraWorker`` 号称"采集与识别解耦"，实际上
   ``engine.predict(frame)`` 就写在采集循环里同步执行（deep 引擎约 200ms/张，
   采集帧率会被拖到 ~5fps），docstring 与 README 的"画面流畅不掉帧"并不成立；
   同时 ``_recognizing`` 标志只在 ``__init__`` 赋值过一次、从此再没被读过，
   是一段死逻辑。现在改为真的双线程：采集循环里**没有任何模型推理**。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..config import Config
from ..engines import INFERENCE_LOCK, get_engine
from ..types import PlateResult
from ..utils.image import imread
from ..utils.logging_setup import get_logger
from ..utils.security import Limits, check_file, check_image

logger = get_logger("ui.worker")


class ImageWorker(QThread):
    """对单张图片执行识别。"""

    finished_with_results = Signal(str, list, float)  # path, results, 耗时ms
    failed = Signal(str, str)

    def __init__(
        self,
        engine_name: str,
        config: Config,
        limits: Limits | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine_name = engine_name
        self.config = config
        #: 安全限制；不传则回落到 config.security（与 Recognizer 行为一致）
        self.limits = limits if limits is not None else config.security.to_limits()
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

    def _load(self) -> np.ndarray | None:
        """按安全策略取图（校验文件大小/扩展名/像素数）。"""
        if self.image_array is not None:
            check_image(self.image_array, self.limits)
            return self.image_array
        if not self.image_path:
            return None
        path = check_file(self.image_path, self.limits)
        img = imread(path)
        check_image(img, self.limits)
        return img

    def run(self) -> None:
        try:
            engine = get_engine(self.engine_name, self.config.engine)
            started = time.perf_counter()
            img = self._load()
            if img is None:
                results: list[PlateResult] = []
            else:
                with INFERENCE_LOCK:
                    results = engine.predict(img)

            elapsed = (time.perf_counter() - started) * 1000
            self.finished_with_results.emit(str(self.image_path), results, elapsed)
        except Exception as exc:
            logger.exception("识别失败")
            self.failed.emit(str(self.image_path), str(exc))


class _RecognizeThread(QThread):
    """独立识别线程：只对"最新一帧"推理，绝不阻塞采集。

    取帧用"序号 + 最新引用"的方式：``_source`` 返回 ``(frame, seq)``，
    只要 ``seq`` 变了说明采集到了新帧，就丢掉手上没跑完的旧帧、直接认新帧。
    这样即使识别慢于采集，也永远是在**最新画面**上推理，不会积压。
    """

    results_ready = Signal(list, float)  # results, 耗时ms
    recognition_error = Signal(str)

    def __init__(
        self,
        engine_name: str,
        config: Config,
        interval: float,
        limits: Limits | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine_name = engine_name
        self.config = config
        self.interval = max(0.0, interval)
        self.limits = limits if limits is not None else config.security.to_limits()
        self._running = False
        self._source = None  # type: ignore[assignment]
        self._done_seq = -1
        self._last_start = 0.0

    def bind_source(self, source) -> None:
        """绑定取帧回调 ``() -> (frame, seq) | None``。"""
        self._source = source

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running:
            if self._source is None:
                self.msleep(20)
                continue

            item = self._source()
            if item is None:
                self.msleep(20)
                continue

            frame, seq = item
            now = time.monotonic()
            # 同一帧不重复识别；且两次识别之间至少间隔 interval（限流防打满 CPU）
            if seq == self._done_seq or (now - self._last_start) < self.interval:
                self.msleep(15)
                continue

            self._done_seq = seq
            self._last_start = now
            try:
                check_image(frame, self.limits)
                engine = get_engine(self.engine_name, self.config.engine)
                started = time.perf_counter()
                with INFERENCE_LOCK:
                    results = engine.predict(frame)
                elapsed = (time.perf_counter() - started) * 1000
                self.results_ready.emit(results, elapsed)
            except Exception as exc:
                logger.warning("视频帧识别失败: %s", exc)
                self.recognition_error.emit(f"识别出错: {exc}")
                self.msleep(50)


class CameraWorker(QThread):
    """摄像头采集线程（**只采集，不识别**）。

    采集循环里绝不调用模型；识别交给内部的 :class:`_RecognizeThread`。
    因此即便单次识别耗时 200ms，主窗口画面仍以摄像头原生帧率刷新。
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
        limits: Limits | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine_name = engine_name
        self.config = config
        self.camera_index = camera_index
        self.interval = max(0.05, interval)
        self.resolution = resolution
        self.limits = limits if limits is not None else config.security.to_limits()
        self._running = False
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._snap_lock = threading.Lock()
        self._capture_fps = 0.0
        # 在主线程里创建识别线程（对象亲和主线程，是标准 Qt 用法），
        # 真正跑起来在 CameraWorker.run() 里 start()。
        self._recog = _RecognizeThread(
            engine_name,
            config,
            self.interval,
            limits=self.limits,
        )
        self._recog.bind_source(self._snapshot)
        self._recog.results_ready.connect(self._on_recog_results)
        self._recog.recognition_error.connect(self._on_recog_error)

    # ------------------------------------------------------------ 子线程信号转发

    def _on_recog_results(self, results: list, elapsed: float) -> None:
        self.results_ready.emit(results, elapsed)

    def _on_recog_error(self, message: str) -> None:
        self.camera_error.emit(message)

    # ------------------------------------------------------------ 生命周期

    def stop(self) -> None:
        """请求停止（线程安全）。采集线程与识别线程一起停。"""
        self._running = False
        self._recog.stop()

    def wait(self, msecs: int = 30000) -> bool:  # type: ignore[override]
        """等待采集线程退出，并顺带收掉识别线程。

        识别线程可能正卡在一次推理里（deep 引擎约 200ms），这里给它
        单独的等待预算，避免主线程关窗时被拖住。
        """
        ok = super().wait(msecs)
        self._recog.stop()
        if not self._recog.wait(msecs):
            logger.warning("识别线程未在 %dms 内退出，强制终止", msecs)
            self._recog.terminate()
            self._recog.wait(500)
        return ok

    # ------------------------------------------------------------ 取帧

    def _snapshot(self) -> tuple[np.ndarray, int] | None:
        """给识别线程取"最新帧 + 序号"（帧对象每次都是新的，无需深拷贝）。"""
        with self._snap_lock:
            if self._latest is None:
                return None
            return self._latest, self._seq

    def grab_latest(self) -> np.ndarray | None:
        """取当前帧的拷贝（抓拍用）。"""
        snap = self._snapshot()
        return None if snap is None else snap[0].copy()

    @property
    def capture_fps(self) -> float:
        """采集帧率（实测值），用于确认"识别没有拖慢画面"。"""
        return self._capture_fps

    # ------------------------------------------------------------ 主循环

    def run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.camera_error.emit(f"无法打开摄像头 {self.camera_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # 先起识别线程，再进采集循环 —— 采集循环里绝不会出现推理
        self._recog.start()

        self._running = True
        read_fail = 0
        frames = 0
        fps_window_start = time.monotonic()
        logger.info("摄像头已启动 index=%d, 分辨率请求 %s", self.camera_index, self.resolution)

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    read_fail += 1
                    # 偶发读帧失败不退出；连续失败才报错（原版直接退出）
                    if read_fail >= 30:
                        self.camera_error.emit("摄像头连续读取失败，已停止")
                        break
                    time.sleep(0.02)
                    continue
                read_fail = 0

                with self._snap_lock:
                    self._latest = frame
                    self._seq += 1
                self.frame_ready.emit(frame)

                frames += 1
                elapsed = time.monotonic() - fps_window_start
                if elapsed >= 1.0:
                    self._capture_fps = frames / elapsed
                    frames = 0
                    fps_window_start = time.monotonic()
        finally:
            self._recog.stop()
            self._recog.wait(1500)
            cap.release()
            logger.info("摄像头已释放")


__all__ = ["ImageWorker", "CameraWorker"]
