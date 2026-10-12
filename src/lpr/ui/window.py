"""主窗口。

相对原版 tkinter 界面（``surface.py``，约 100 行）的升级：

* **PySide6 + QSS 主题**：现代观感，DPI 自适应（tkinter 在高分屏发虚）。
* **批量识别**：原版一次只能选一张图，现在可多选、逐个队列识别。
* **实时画面不再冻结**：识别在 ``CameraWorker`` 里做，采集帧率不受影响；
  原版"每 2 秒卡一下"的根源就是采集和识别没分开。
* **历史记录 + CSV 导出**：原版识别完就丢了，没有记录。
* **引擎热切换**：下拉框切引擎，立即生效；不可用的引擎自动置灰并说明原因。
* **阈值/参数可视化**：把最常调的几个定位参数暴露到设置面板，
  不用再改 config.js + 重启。
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from ..engines import describe_engines
from ..types import PlateColor, PlateResult
from ..utils.image import imread, to_display_rgb
from ..utils.logging_setup import get_logger, memory_handler
from ..utils.security import (
    PrivacyPolicy,
    SecurityError,
    check_file,
    check_image,
    mask_plate,
)
from .worker import CameraWorker, ImageWorker, list_cameras

logger = get_logger("ui.window")

#: 颜色标签 → UI 颜色（RGB hex）
_LABEL_HEX = {
    PlateColor.BLACK.value: "#3d4451",
    PlateColor.BLUE.value: "#1e6fd9",
    PlateColor.GREEN.value: "#1f9d6b",
    PlateColor.WHITE.value: "#6b7688",
    PlateColor.YELLOW.value: "#d99a06",
    PlateColor.HAZMAT.value: "#d64545",
    PlateColor.UNKNOWN.value: "#8a93a3",
}

# 设计令牌：整套界面只用这几个颜色/圆角，改主题只动这里
_T = {
    "bg": "#eef1f6",  # 窗口底色
    "surface": "#ffffff",  # 卡片
    "border": "#e2e7ef",  # 卡片描边
    "text": "#1f2733",  # 正文
    "muted": "#7b8499",  # 次要文字
    "accent": "#2f6bed",  # 主色
    "accent_hover": "#245fcf",
    "danger": "#e5484d",
}

_QSS = f"""
QMainWindow, QMenuBar {{ background: {_T["bg"]}; }}
QMenuBar {{ padding: 2px 8px; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: 5px; color: {_T["text"]}; }}
QMenuBar::item:selected {{ background: #dfe5ee; }}
QMenu {{ background: white; border: 1px solid {_T["border"]}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 24px; border-radius: 5px; }}
QMenu::item:selected {{ background: {_T["accent"]}; color: white; }}

QGroupBox {{
    font-weight: 600; font-size: 13px;
    border: 1px solid {_T["border"]}; border-radius: 10px;
    margin-top: 14px; background: {_T["surface"]}; padding: 8px 10px 10px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 14px; padding: 0 6px;
    color: {_T["muted"]};
}}

QPushButton {{
    background: {_T["accent"]}; color: white; border: none; border-radius: 8px;
    padding: 0 18px; min-height: 32px; font-size: 13px; font-weight: 500;
}}
QPushButton:hover {{ background: {_T["accent_hover"]}; }}
QPushButton:pressed {{ background: #1d4fae; }}
QPushButton:disabled {{ background: #c3ccda; }}
QPushButton#secondary {{
    background: white; color: {_T["text"]}; border: 1px solid #cdd5e1;
}}
QPushButton#secondary:hover {{ background: #f2f5fa; border-color: {_T["accent"]}; }}
QPushButton#secondary:pressed {{ background: #e6ecf5; }}
QPushButton#danger {{ background: {_T["danger"]}; }}
QPushButton#danger:hover {{ background: #c93a3f; }}

QComboBox {{
    border: 1px solid #cdd5e1; border-radius: 8px; padding: 0 10px;
    min-height: 32px; background: white; color: {_T["text"]};
}}
QComboBox:hover {{ border-color: {_T["accent"]}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: white; border: 1px solid {_T["border"]}; border-radius: 8px;
    selection-background-color: #e3ecff; selection-color: {_T["text"]};
    outline: none; padding: 4px;
}}

QTableWidget {{
    border: none; background: {_T["surface"]};
    gridline-color: #eef1f6; selection-background-color: #e3ecff;
    selection-color: {_T["text"]}; alternate-background-color: #f8fafd;
}}
QHeaderView::section {{
    background: #f4f6fa; color: {_T["muted"]}; border: none;
    border-bottom: 1px solid {_T["border"]}; padding: 7px 6px; font-weight: 600;
}}
QTableCornerButton::section {{ background: #f4f6fa; border: none; }}

QLabel#imageArea {{
    border: 2px dashed #c9d2e0; border-radius: 12px; background: #fbfcfe;
    color: {_T["muted"]}; font-size: 14px;
}}
QLabel#plateCard {{
    background: #242b38; border-radius: 10px;
}}
QLabel#colorBadge {{
    border-radius: 10px; font-weight: 600; font-size: 15px;
}}
QLabel#metaLabel {{ color: {_T["muted"]}; font-size: 12px; }}
QLabel#hintLabel {{ color: {_T["muted"]}; font-size: 12px; }}

QProgressBar {{
    background: #e3e8f0; border: none; border-radius: 7px;
    min-height: 14px; max-height: 14px; text-align: center;
    color: transparent; font-size: 9px;
}}
QProgressBar::chunk {{ background: {_T["accent"]}; border-radius: 7px; }}

QScrollBar:vertical {{ background: transparent; width: 9px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #c9d1de; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #aab5c6; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #c9d1de; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {_T["bg"]}; width: 10px; }}
QToolTip {{
    background: #2b3342; color: white; border: none; border-radius: 6px; padding: 6px 10px;
}}
"""


def _array_to_pixmap(img_bgr: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    """BGR 数组 → QPixmap，按容器缩放。"""
    rgb = to_display_rgb(img_bgr)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class MainWindow(QMainWindow):
    """主窗口。"""

    #: 后台探测线程 -> 主线程：可用摄像头索引列表（跨线程，自动队列投递）
    cameras_found = Signal(list)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        #: 安全策略：与 CLI / Python API 走同一份 config.security（v2.3.1 修复）。
        #: 此前界面层完全没有接入安全层 —— 图片不做大小/像素校验、CSV 导出
        #: 明文写车牌号、日志脱敏过滤器也没装，README 的「更安全」在 GUI 上不成立。
        self._security = config.security.to_context()
        self._security.setup()
        self.limits = self._security.limits
        self.privacy: PrivacyPolicy = self._security.privacy
        self._worker: ImageWorker | None = None
        self._camera: CameraWorker | None = None
        self._history: list[PlateResult] = []
        self._batch_queue: list[Path] = []
        self._batch_done = 0
        self._batch_total = 0
        self._last_roi_pixmap: QPixmap | None = None

        self.setWindowTitle(config.ui.window_title)
        self.resize(*config.ui.window_size)
        self.setStyleSheet(_QSS)

        self._build_menu()
        self._build_ui()
        self._refresh_engine_list()
        self._refresh_log_view()
        if self.privacy.mask_output:
            logger.info("隐私策略已启用：导出的 CSV 中车牌号将被打码")

        # 日志面板定时刷新
        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._refresh_log_view)
        self._log_timer.start(1500)

        # 后台探测可用摄像头，填充下拉框（逐个打开设备可能各花数百毫秒，
        # 放在 UI 线程会让窗口启动卡顿）
        self.cameras_found.connect(self._on_cameras_found)
        threading.Thread(target=self._probe_cameras, daemon=True).start()

    # ---------------------------------------------------------------- 菜单

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("文件(&F)")

        act_open = QAction("打开图片(&O)...", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.on_open_images)
        menu.addAction(act_open)

        act_export = QAction("导出历史记录(&E)...", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self.on_export_history)
        menu.addAction(act_export)

        menu.addSeparator()
        act_quit = QAction("退出(&Q)", self)
        act_quit.triggered.connect(self.close)
        menu.addAction(act_quit)

    # ---------------------------------------------------------------- 界面

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---------- 左侧：工具条 + 图像显示 ----------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 顶部工具条（操作在画面上方，符合自上而下的使用顺序）
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self.on_open_images)
        self.btn_batch = QPushButton("批量识别文件夹")
        self.btn_batch.setObjectName("secondary")
        self.btn_batch.clicked.connect(self.on_batch_folder)
        self.btn_camera = QPushButton("启动摄像头")
        self.btn_camera.setObjectName("secondary")
        self.btn_camera.clicked.connect(self.on_toggle_camera)
        # 多摄像头机器上索引 0 未必是用户想要的那个；默认探测一遍填进来
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip("要使用的摄像头索引（启动前自动探测）")
        camera_caption = QLabel("摄像头")
        camera_caption.setStyleSheet(f"color:{_T['muted']};")
        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_batch)
        toolbar.addWidget(self.btn_camera)
        toolbar.addSpacing(6)
        toolbar.addWidget(camera_caption)
        toolbar.addWidget(self.camera_combo)
        toolbar.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        toolbar.addWidget(self.progress)
        left_layout.addLayout(toolbar)

        self.image_label = QLabel("打开图片或启动摄像头\n（Ctrl+O）")
        self.image_label.setObjectName("imageArea")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(560, 420)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)
        left_layout.addWidget(self.image_label, 1)

        splitter.addWidget(left)

        # ---------- 右侧：结果面板 ----------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 引擎选择
        engine_box = QGroupBox("识别引擎")
        engine_layout = QGridLayout(engine_box)
        self.engine_combo = QComboBox()
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        self.engine_hint = QLabel("")
        self.engine_hint.setWordWrap(True)
        self.engine_hint.setObjectName("hintLabel")
        engine_layout.addWidget(QLabel("引擎"), 0, 0)
        engine_layout.addWidget(self.engine_combo, 0, 1)
        engine_layout.addWidget(self.engine_hint, 1, 0, 1, 2)
        right_layout.addWidget(engine_box)

        # 识别结果
        result_box = QGroupBox("识别结果")
        result_layout = QGridLayout(result_box)
        result_layout.setSpacing(10)

        # 车牌号放在深色展示卡上（白底大字刺眼，深底上各牌色都清楚）
        self.plate_label = QLabel("—")
        self.plate_label.setObjectName("plateCard")
        plate_font = QFont()
        plate_font.setPointSize(24)
        plate_font.setBold(True)
        plate_font.setLetterSpacing(QFont.AbsoluteSpacing, 4.0)
        self.plate_label.setFont(plate_font)
        self.plate_label.setAlignment(Qt.AlignCenter)
        self.plate_label.setMinimumHeight(64)
        self.plate_label.setStyleSheet(f"color:{_LABEL_HEX[PlateColor.BLUE.value]};")

        self.roi_label = QLabel()
        self.roi_label.setAlignment(Qt.AlignCenter)
        self.roi_label.setMinimumSize(240, 88)
        self.roi_label.setStyleSheet("background:#f4f6fa; border-radius:8px;")

        self.meta_label = QLabel("等待识别…")
        self.meta_label.setWordWrap(True)
        self.meta_label.setObjectName("metaLabel")

        result_layout.addWidget(self.plate_label, 0, 0, 1, 2)
        result_layout.addWidget(self.roi_label, 1, 0)
        color_card = QLabel("—")
        color_card.setObjectName("colorBadge")
        color_card.setAlignment(Qt.AlignCenter)
        self.color_card = color_card
        result_layout.addWidget(color_card, 1, 1)
        result_layout.addWidget(self.meta_label, 2, 0, 1, 2)
        right_layout.addWidget(result_box)

        # 历史记录
        history_box = QGroupBox("识别历史")
        history_layout = QVBoxLayout(history_box)
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(["时间", "号码", "颜色", "置信度", "引擎"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setShowGrid(False)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.verticalHeader().setDefaultSectionSize(30)
        self.history_table.setMaximumHeight(220)
        history_layout.addWidget(self.history_table)

        btn_row = QHBoxLayout()
        btn_export = QPushButton("导出 CSV")
        btn_export.setObjectName("secondary")
        btn_export.clicked.connect(self.on_export_history)
        btn_clear = QPushButton("清空历史")
        btn_clear.setObjectName("danger")
        btn_clear.clicked.connect(self.on_clear_history)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        history_layout.addLayout(btn_row)
        right_layout.addWidget(history_box)

        # 运行日志
        log_box = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QTableWidget(0, 1)
        self.log_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log_view.verticalHeader().setVisible(False)
        self.log_view.setEditTriggers(QTableWidget.NoEditTriggers)
        self.log_view.setShowGrid(False)
        self.log_view.setAlternatingRowColors(True)
        self.log_view.verticalHeader().setDefaultSectionSize(26)
        log_font = QFont("Consolas")
        log_font.setStyleHint(QFont.Monospace)
        log_font.setPointSize(9)
        self.log_view.setFont(log_font)
        self.log_view.setMaximumHeight(160)
        log_layout.addWidget(self.log_view)
        right_layout.addWidget(log_box)

        splitter.addWidget(right)
        splitter.setSizes([720, 480])

    # ---------------------------------------------------------------- 引擎

    def _refresh_engine_list(self) -> None:
        self.engine_combo.blockSignals(True)
        self.engine_combo.clear()
        infos = describe_engines()
        default_idx = 0
        for i, info in enumerate(infos):
            label = info["display_name"]
            if not info["available"]:
                label += "（不可用）"
            self.engine_combo.addItem(label, info["name"])
            if not info["available"]:
                # Qt 下拉框单项禁用需要 model API
                from PySide6.QtGui import QStandardItemModel

                model = self.engine_combo.model()
                if isinstance(model, QStandardItemModel):
                    item = model.item(i)
                    item.setEnabled(False)
            elif info["name"] == self.config.engine.default:
                default_idx = i
        self.engine_combo.setCurrentIndex(default_idx)
        self.engine_combo.blockSignals(False)
        self._update_engine_hint()

    def _update_engine_hint(self) -> None:
        name = self.engine_combo.currentData()
        for info in describe_engines():
            if info["name"] == name:
                self.engine_hint.setText(
                    info["description"] if info["available"] else f"不可用: {info['reason']}"
                )
                break

    def on_engine_changed(self) -> None:
        name = self.engine_combo.currentData()
        if not name:
            return
        self.config.engine.default = name
        self._update_engine_hint()
        logger.info("切换引擎 → %s", name)

    # ---------------------------------------------------------------- 图片识别

    def on_open_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择识别图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)"
        )
        if not paths:
            return
        self._start_batch([Path(p) for p in paths])

    def on_batch_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        root = Path(folder)
        files = sorted(
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in self.limits.allowed_extensions
        )
        if not files:
            QMessageBox.information(self, "提示", "该文件夹中没有图片文件。")
            return
        self._start_batch(files)

    def _start_batch(self, files: list[Path]) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "识别正在进行中，请稍候。")
            return
        if not files:
            return

        # 与 CLI 一致地套用批量文件数上限，避免一次塞入过多文件
        limit = self.limits.max_batch_files
        if len(files) > limit:
            logger.warning("批量任务超过上限 %d，已截断", limit)
            QMessageBox.information(
                self, "提示", f"选择 {len(files)} 张，超过上限 {limit} 张，已截断。"
            )
            files = files[:limit]

        self._batch_queue = list(files)
        self._batch_total = len(files)
        self._batch_done = 0
        self.progress.setVisible(True)
        self.progress.setRange(0, self._batch_total)
        self.progress.setValue(0)
        self._next_in_queue()

    def _next_in_queue(self) -> None:
        if not self._batch_queue:
            self.progress.setVisible(False)
            logger.info("批量识别完成，共 %d 张", self._batch_total)
            return

        path = self._batch_queue.pop(0)
        try:
            # 走安全层：校验扩展名白名单、文件大小上限后再解码
            img = imread(check_file(path, self.limits))
            check_image(img, self.limits)
        except (SecurityError, OSError, ValueError) as exc:
            logger.warning("跳过 %s: %s", path, exc)
            self._batch_done += 1
            self.progress.setValue(self._batch_done)
            self._next_in_queue()
            return

        # 显示原图
        self._show_image(img)
        self.setWindowTitle(f"{self.config.ui.window_title} - {path.name}")

        self._worker = ImageWorker(
            self.config.engine.default, self.config, limits=self.limits, parent=self
        )
        self._worker.set_array(img, source=str(path))
        self._worker.finished_with_results.connect(self._on_image_done)
        self._worker.failed.connect(self._on_image_failed)
        self._worker.start()

    def _on_image_done(self, source: str, results: list, elapsed: float) -> None:
        self._show_results(results, elapsed, source)
        self._batch_done += 1
        self.progress.setValue(self._batch_done)
        self._worker = None
        QTimer.singleShot(30, self._next_in_queue)  # 让 UI 呼吸一下

    def _on_image_failed(self, source: str, error: str) -> None:
        logger.error("识别失败 %s: %s", source, error)
        QMessageBox.critical(self, "识别失败", f"{source}\n\n{error}")
        self._batch_done += 1
        self.progress.setValue(self._batch_done)
        self._worker = None
        QTimer.singleShot(30, self._next_in_queue)

    def _show_image(self, img: np.ndarray) -> None:
        w = max(320, self.image_label.width() - 8)
        h = max(240, self.image_label.height() - 8)
        self.image_label.setPixmap(_array_to_pixmap(img, w, h))
        self._current_image = img

    def _show_results(self, results: list[PlateResult], elapsed: float, source: str) -> None:
        if not results:
            self.plate_label.setText("未识别到车牌")
            self.plate_label.setStyleSheet("color:#e5484d;")  # 深色卡片上红字更醒目
            self.color_card.setText("—")
            self.meta_label.setText(f"耗时 {elapsed:.0f} ms\n{source}")
            return

        best = results[0]
        self.plate_label.setText(best.text)
        self.plate_label.setStyleSheet(f"color:{_LABEL_HEX.get(best.color.value, '#e8ecf4')};")
        self.color_card.setText(best.color.label)
        hex_color = _LABEL_HEX.get(best.color.value, "#8a93a3")
        # Qt QSS 不支持 #RRGGBBAA 八位写法（会解析失败渲染成怪色），
        # 半透明底色必须用 rgba() 函数形式
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
        self.color_card.setStyleSheet(
            f"background:rgba({r}, {g}, {b}, 0.14);"
            f"border:1px solid rgba({r}, {g}, {b}, 0.55);"
            f"color:{hex_color};"
        )

        if best.roi is not None and best.roi.size:
            self._last_roi_pixmap = _array_to_pixmap(
                best.roi, self.roi_label.width(), self.roi_label.height()
            )
            self.roi_label.setPixmap(self._last_roi_pixmap)

        chars_info = (
            " ".join(f"{t}:{c:.2f}" for t, c in zip(best.text, best.char_confidences, strict=False))
            if best.char_confidences
            else ""
        )
        self.meta_label.setText(
            f"耗时 {elapsed:.0f} ms｜引擎 {best.engine}｜候选 {len(results)} 个\n"
            f"bbox={best.bbox}  {chars_info}"
        )

        if len(results) > 1:
            logger.info("共 %d 个车牌: %s", len(results), " / ".join(r.text for r in results))

        self._append_history(best)

    # ---------------------------------------------------------------- 摄像头

    def _probe_cameras(self) -> None:
        """在后台线程里枚举可用摄像头（纯 cv2 操作，不碰任何 UI 对象）。"""
        try:
            found = list_cameras()
        except Exception:
            logger.warning("摄像头探测失败", exc_info=True)
            found = []
        self.cameras_found.emit(found)

    def _on_cameras_found(self, found: list) -> None:
        self.camera_combo.clear()
        if not found:
            # 一个都没有：仍然给个默认 0，让按钮可点、错误提示可触发
            self.camera_combo.addItem("未检测到摄像头", 0)
            self.camera_combo.setEnabled(False)
            return
        for idx in found:
            self.camera_combo.addItem(f"摄像头 {idx}", idx)
        self.camera_combo.setEnabled(True)

    def on_toggle_camera(self) -> None:
        if self._camera is not None and self._camera.isRunning():
            self._stop_camera()
            return

        # 探测尚未完成时 currentData() 为 None，回落到默认索引 0
        index = self.camera_combo.currentData()
        self._camera = CameraWorker(
            self.config.engine.default,
            self.config,
            camera_index=index if isinstance(index, int) else 0,
            interval=self.config.ui.camera_interval,
            resolution=self.config.ui.camera_resolution,
            limits=self.limits,
        )
        self._camera.frame_ready.connect(self._on_camera_frame)
        self._camera.results_ready.connect(self._on_camera_results)
        self._camera.camera_error.connect(self._on_camera_error)
        self._camera.start()
        self.btn_camera.setText("停止摄像头")
        logger.info("摄像头线程已启动")

    def _stop_camera(self) -> None:
        if self._camera is None:
            return
        self._camera.stop()
        # 给线程最多 2 秒优雅退出
        if not self._camera.wait(2000):
            logger.warning("摄像头线程未在 2s 内退出，强制终止")
            self._camera.terminate()
        self._camera = None
        self.btn_camera.setText("启动摄像头")
        self.image_label.clear()
        self.image_label.setText("打开图片或启动摄像头\n（Ctrl+O）")

    def _on_camera_frame(self, frame) -> None:
        w = max(320, self.image_label.width() - 8)
        h = max(240, self.image_label.height() - 8)
        self.image_label.setPixmap(_array_to_pixmap(frame, w, h))

    def _on_camera_results(self, results: list, elapsed: float) -> None:
        if results:
            self._show_results(results, elapsed, "<camera>")
        else:
            # 清掉上一次的结果但不闪烁
            self.plate_label.setText("识别中…")
            self.color_card.setText("")

    def _on_camera_error(self, message: str) -> None:
        QMessageBox.warning(self, "摄像头", message)
        self._stop_camera()

    def closeEvent(self, event) -> None:
        """关窗时优雅停掉所有线程（原版这里会超时强杀）。"""
        self._log_timer.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        if self._camera is not None:
            self._camera.stop()
            self._camera.wait(2000)
        super().closeEvent(event)

    # ---------------------------------------------------------------- 历史

    def _append_history(self, res: PlateResult) -> None:
        self._history.append(res)
        if len(self._history) > self.config.ui.max_history:
            self._history = self._history[-self.config.ui.max_history :]

        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        values = [
            datetime.now().strftime("%H:%M:%S"),
            res.text,
            res.color.label,
            f"{res.confidence:.2f}",
            res.engine,
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if col == 1:
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            self.history_table.setItem(row, col, item)
        self.history_table.scrollToBottom()

    def on_export_history(self) -> None:
        if not self._history:
            QMessageBox.information(self, "提示", "没有可导出的历史记录。")
            return
        default_name = f"lpr_history_{datetime.now():%Y%m%d_%H%M%S}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", default_name, "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.writer(fp)
            writer.writerow(["时间", "号码", "颜色", "置信度", "引擎", "bbox"])
            for res in self._history:
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        # 导出是"输出"，受 privacy 策略约束（此前直接写明文）
                        mask_plate(res.text, self.privacy),
                        res.color.label,
                        f"{res.confidence:.4f}",
                        res.engine,
                        str(res.bbox),
                    ]
                )
        hint = "（车牌号已按隐私策略打码）" if self.privacy.mask_output else ""
        QMessageBox.information(self, "完成", f"已导出 {len(self._history)} 条{hint}到\n{path}")

    def on_clear_history(self) -> None:
        self._history.clear()
        self.history_table.setRowCount(0)

    # ---------------------------------------------------------------- 日志

    def _refresh_log_view(self) -> None:
        records = memory_handler.dump()[-20:]
        if self.log_view.rowCount() == len(records):
            return
        self.log_view.setRowCount(0)
        for line in records:
            row = self.log_view.rowCount()
            self.log_view.insertRow(row)
            self.log_view.setItem(row, 0, QTableWidgetItem(line))
        self.log_view.scrollToBottom()


def run_ui(config: Config | None = None) -> None:
    """启动界面（独立入口）。"""
    import sys

    from PySide6.QtWidgets import QApplication

    config = config or Config.load()
    app = QApplication(sys.argv)
    app.setApplicationName("车牌识别系统 v2")

    win = MainWindow(config)
    win.show()
    sys.exit(app.exec())


__all__ = ["MainWindow", "run_ui"]
