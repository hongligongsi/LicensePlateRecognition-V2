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
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGraphicsDropShadowEffect,
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

from .. import __version__
from ..config import Config
from ..engines import describe_engines
from ..types import PlateColor, PlateResult
from ..utils.image import imread, to_display_rgb
from ..utils.logging_setup import (
    current_log_file,
    get_logger,
    memory_handler,
    setup_logging,
)
from ..utils.security import (
    PrivacyPolicy,
    SecurityError,
    check_file,
    check_image,
    mask_plate,
)
from .worker import CameraWorker, ImageWorker, list_cameras

logger = get_logger("ui.window")


def _open_directory(path: Path) -> str | None:
    """用系统文件管理器打开目录，成功返回 ``None``，失败返回原因文本。

    三平台各一条命令。任何异常都降级为"把路径显示给用户"而不是抛出去 ——
    打不开文件管理器是小事，把界面弄崩是大事。
    """
    try:
        if sys.platform.startswith("win"):
            # Windows 上 os 有 startfile，Linux/macOS 的类型桩没有（attr-defined 报错）；
            # 同时列出 unused-ignore 兼容两平台：只屏蔽自己这一行的"未使用"告警。
            os.startfile(str(path))  # type: ignore[attr-defined, unused-ignore]
            return None
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        return str(exc)


#: 颜色标签 → 文本色（用于历史表格等浅底场景）
_LABEL_HEX = {
    PlateColor.BLACK.value: "#3d4451",
    PlateColor.BLUE.value: "#1e6fd9",
    PlateColor.GREEN.value: "#1f9d6b",
    PlateColor.WHITE.value: "#6b7688",
    PlateColor.YELLOW.value: "#d99a06",
    PlateColor.HAZMAT.value: "#d64545",
    PlateColor.UNKNOWN.value: "#8a93a3",
}

#: 颜色标签 → 车牌"皮肤"（底/字/描边）。
#: 按国标配色渲染：蓝牌蓝底白字、黄牌黄底黑字、新能源绿底黑字、
#: 白牌（军/警/使领）白底黑字、黑牌（使领馆）黑底白字。
#: 结果区直接把号码画成车牌的样子，比单纯染个字色信息量大得多。
_PLATE_SKIN: dict[str, dict[str, str]] = {
    PlateColor.BLUE.value: {
        "bg": "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2f74dd, stop:1 #1a4fa8)",
        "fg": "#ffffff",
        "border": "#143f86",
    },
    PlateColor.YELLOW.value: {
        "bg": "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fad04f, stop:1 #e2a800)",
        "fg": "#1a1a1a",
        "border": "#b58400",
    },
    PlateColor.GREEN.value: {
        "bg": "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8fe0a4, stop:1 #35a566)",
        "fg": "#0d2b1a",
        "border": "#1f7a45",
    },
    PlateColor.WHITE.value: {
        "bg": "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e8ecf3)",
        "fg": "#1b1f27",
        "border": "#b9c2d0",
    },
    PlateColor.BLACK.value: {
        "bg": "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2c3442, stop:1 #171c25)",
        "fg": "#f2f5fa",
        "border": "#454f61",
    },
    PlateColor.HAZMAT.value: {
        "bg": "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e8ecf3)",
        "fg": "#c0392b",
        "border": "#b9c2d0",
    },
    PlateColor.UNKNOWN.value: {
        "bg": "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #39424f, stop:1 #242b36)",
        "fg": "#cbd3e0",
        "border": "#4a5567",
    },
}

#: 颜色标签 → 徽章用色（浅底胶囊）
_BADGE_HEX = {
    PlateColor.BLUE.value: "#1e6fd9",
    PlateColor.YELLOW.value: "#b8860b",
    PlateColor.GREEN.value: "#1f9d6b",
    PlateColor.WHITE.value: "#6b7688",
    PlateColor.BLACK.value: "#3d4451",
    PlateColor.HAZMAT.value: "#c0392b",
    PlateColor.UNKNOWN.value: "#8a93a3",
}

#: 设计令牌：整套界面只用这几个颜色。两套主题共用同一组键名，
#: 加一套主题 = 加一个 dict —— 控件代码一行都不用改（见 :func:`build_qss`）。
#: 车牌自身的国标配色**不在这里**（那是物理属性，与界面明暗无关，见 ``_PLATE_SKIN``）。
_THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#eef1f6",  # 窗口底色
        "surface": "#ffffff",  # 卡片
        "surface2": "#f4f6fa",  # 表头 / 次级面板
        "border": "#e2e7ef",  # 卡片描边
        "text": "#1f2733",  # 正文
        "muted": "#7b8499",  # 次要文字
        "accent": "#2f6bed",  # 主色
        "accent_hover": "#245fcf",
        "accent_pressed": "#1d4fae",
        "accent_soft": "#e3ecff",  # 选中态浅底
        "disabled": "#c3ccda",
        "danger": "#e5484d",
        "danger_hover": "#c93a3f",
        "ctrl_border": "#cdd5e1",  # 次级按钮/输入框描边
        "ctrl_hover": "#f2f5fa",
        "ctrl_pressed": "#e6ecf5",
        "grid": "#eef1f6",  # 表格网格线
        "alt_row": "#f8fafd",  # 斑马纹
        "image_bg": "#171b22",  # 图像区（两套主题都深色：照片在深底上最清楚）
        "image_border": "#2a323f",
        "progress_bg": "#e3e8f0",
        "scroll": "#c9d1de",
        "scroll_hover": "#aab5c6",
        "tooltip_bg": "#2b3342",
        "shadow": "31, 39, 51, 26",  # 卡片投影 rgba 分量
    },
    "dark": {
        "bg": "#12161d",
        "surface": "#1a2029",
        "surface2": "#212936",
        "border": "#2b3441",
        "text": "#e6ebf2",
        "muted": "#8f9aad",
        "accent": "#4c8dff",
        "accent_hover": "#5f9bff",
        "accent_pressed": "#3d7cf0",
        "accent_soft": "#25354e",
        "disabled": "#39424f",
        "danger": "#ef5350",
        "danger_hover": "#d94448",
        "ctrl_border": "#3a4553",
        "ctrl_hover": "#232c39",
        "ctrl_pressed": "#1e2632",
        "grid": "#232b38",
        "alt_row": "#1e2530",
        "image_bg": "#0c0f14",
        "image_border": "#242c38",
        "progress_bg": "#232b38",
        "scroll": "#39424f",
        "scroll_hover": "#4a5563",
        "tooltip_bg": "#0f131a",
        "shadow": "0, 0, 0, 110",
    },
}

#: 主题名 → 中文显示名（菜单与状态栏用）
THEME_LABELS = {"light": "浅色", "dark": "深色"}

#: 默认主题（与 ``UIConfig.theme`` 的默认值一致）
DEFAULT_THEME = "light"

#: 当前主题令牌。进程级可变状态：GUI 只有一扇主窗口，且 ``build_qss``
#: 在切主题时被重新求值，因此不需要把它挂到实例上（挂上去反而要求
#: 每个引用 `_T[...]` 的地方都改成 `self._t[...]`，漏一处就是半截主题）。
_T: dict[str, str] = dict(_THEMES[DEFAULT_THEME])


def normalize_theme(name: str | None) -> str:
    """把任意主题名收敛到受支持的值（未知/空 → 默认主题）。"""
    return name if name in _THEMES else DEFAULT_THEME


def set_theme(name: str | None) -> str:
    """切换当前主题令牌，返回实际生效的主题名。"""
    key = normalize_theme(name)
    _T.clear()
    _T.update(_THEMES[key])
    return key


def build_qss() -> str:
    """按**当前**主题令牌生成全局 QSS。"""
    return f"""
QMainWindow, QMenuBar {{ background: {_T["bg"]}; }}
QMenuBar {{ padding: 2px 8px; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: 5px; color: {_T["text"]}; }}
QMenuBar::item:selected {{ background: {_T["accent_soft"]}; }}
QMenu {{
    background: {_T["surface"]}; border: 1px solid {_T["border"]};
    border-radius: 8px; padding: 4px;
}}
QMenu::item {{ padding: 6px 24px; border-radius: 5px; color: {_T["text"]}; }}
QMenu::item:selected {{ background: {_T["accent"]}; color: white; }}
QMenu::separator {{ height: 1px; background: {_T["border"]}; margin: 4px 8px; }}

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
QPushButton:pressed {{ background: {_T["accent_pressed"]}; }}
QPushButton:disabled {{ background: {_T["disabled"]}; color: {_T["muted"]}; }}
QPushButton#secondary {{
    background: {_T["surface"]}; color: {_T["text"]};
    border: 1px solid {_T["ctrl_border"]};
}}
QPushButton#secondary:hover {{
    background: {_T["ctrl_hover"]}; border-color: {_T["accent"]};
}}
QPushButton#secondary:pressed {{ background: {_T["ctrl_pressed"]}; }}
QPushButton#danger {{ background: {_T["danger"]}; }}
QPushButton#danger:hover {{ background: {_T["danger_hover"]}; }}
/* 小尺寸按钮：复制这类轻量动作不该和主按钮一样抢眼 */
QPushButton#mini {{
    background: {_T["surface"]}; color: {_T["muted"]};
    border: 1px solid {_T["ctrl_border"]}; border-radius: 6px;
    padding: 0 10px; min-height: 24px; max-height: 24px; font-size: 12px;
}}
QPushButton#mini:hover {{ color: {_T["accent"]}; border-color: {_T["accent"]}; }}

QComboBox {{
    border: 1px solid {_T["ctrl_border"]}; border-radius: 8px; padding: 0 10px;
    min-height: 32px; background: {_T["surface"]}; color: {_T["text"]};
}}
QComboBox:hover {{ border-color: {_T["accent"]}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {_T["surface"]}; border: 1px solid {_T["border"]}; border-radius: 8px;
    selection-background-color: {_T["accent_soft"]}; selection-color: {_T["text"]};
    outline: none; padding: 4px;
}}
/* 条目文字必须显式上色：系统深色模式下默认调色板文字是浅色，
   只给弹层设白底不设字色会变成白底白字，条目全部"隐形" */
QComboBox QAbstractItemView::item {{
    background: {_T["surface"]}; color: {_T["text"]};
    min-height: 30px; padding: 4px 8px; border-radius: 6px;
}}
QComboBox QAbstractItemView::item:selected {{
    background: {_T["accent_soft"]}; color: {_T["text"]};
}}
QComboBox QAbstractItemView::item:disabled {{
    background: {_T["surface"]}; color: {_T["muted"]};
}}

QTableWidget {{
    border: none; background: {_T["surface"]};
    gridline-color: {_T["grid"]}; selection-background-color: {_T["accent_soft"]};
    selection-color: {_T["text"]}; alternate-background-color: {_T["alt_row"]};
    color: {_T["text"]};
}}
QHeaderView::section {{
    background: {_T["surface2"]}; color: {_T["muted"]}; border: none;
    border-bottom: 1px solid {_T["border"]}; padding: 7px 6px; font-weight: 600;
}}
QTableCornerButton::section {{ background: {_T["surface2"]}; border: none; }}

QLabel#imageArea {{
    border: 1px solid {_T["image_border"]}; border-radius: 12px; background: {_T["image_bg"]};
    color: #6e7a8d; font-size: 14px;
}}
QLabel#plateCard {{
    border-radius: 12px; padding: 4px 10px;
}}
QLabel#roiArea {{
    background: {_T["surface2"]}; border: 1px solid {_T["border"]}; border-radius: 8px;
    color: {_T["muted"]}; font-size: 12px;
}}
QLabel#colorBadge {{
    border-radius: 10px; font-weight: 600; font-size: 15px;
}}
QLabel#metaLabel {{ color: {_T["muted"]}; font-size: 12px; }}
QLabel#hintLabel {{ color: {_T["muted"]}; font-size: 12px; }}
QWidget#toolCard {{
    background: {_T["surface"]}; border: 1px solid {_T["border"]}; border-radius: 10px;
}}
QLabel#statusChip {{
    color: {_T["muted"]}; font-size: 12px; padding: 0 10px;
}}
QStatusBar {{
    background: {_T["surface"]}; border-top: 1px solid {_T["border"]};
    color: {_T["muted"]}; font-size: 12px;
}}
QStatusBar::item {{ border: none; }}

QProgressBar {{
    background: {_T["progress_bg"]}; border: none; border-radius: 7px;
    min-height: 14px; max-height: 14px; text-align: center;
    color: transparent; font-size: 9px;
}}
QProgressBar::chunk {{ background: {_T["accent"]}; border-radius: 7px; }}

QScrollBar:vertical {{ background: transparent; width: 9px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {_T["scroll"]}; border-radius: 4px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {_T["scroll_hover"]}; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {_T["scroll"]}; border-radius: 4px; min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {_T["scroll_hover"]}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {_T["bg"]}; width: 10px; }}
QToolTip {{
    background: {_T["tooltip_bg"]}; color: white; border: none;
    border-radius: 6px; padding: 6px 10px;
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
        #: 「运行日志」面板上一次渲染的内容快照（判重用；比行数可靠）
        self._log_snapshot: list[str] = []
        #: 最近一次颜色徽章的状态 ``(key, label, muted)``：切主题后要重放
        #: （徽章的浅底胶囊用 rgba 半透明，底色深浅必须跟着主题走）
        self._badge_state: tuple[str, str | None, bool] | None = None
        #: 当前展示的号码（供「复制」按钮取用）
        self._current_text = ""
        #: 当前静态图片（缩放窗口后按新尺寸重绘）
        self._current_image: np.ndarray | None = None
        #: 窗口尺寸变化后的重绘去抖定时器。
        #: **必须在 `resize()` 之前建好** —— resize 会同步触发 resizeEvent，
        #: 那时若它还不存在就会直接抛 AttributeError。
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._redraw_current_image)

        self.setWindowTitle(config.ui.window_title)
        self.resize(*config.ui.window_size)
        #: 外观主题。``UIConfig.theme`` 此前只有字段没有实现（界面永远浅色），
        #: 现在两套令牌共用同一个 `build_qss`，切主题不需要重建任何控件。
        self._theme = set_theme(config.ui.theme)
        self.setStyleSheet(build_qss())
        #: 支持把图片/文件夹直接拖进窗口识别
        self.setAcceptDrops(True)

        self._build_menu()
        self._build_ui()
        self._build_statusbar()
        self._apply_card_shadows()
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

        act_folder = QAction("打开文件夹(&D)...", self)
        act_folder.setShortcut("Ctrl+Shift+O")
        act_folder.triggered.connect(self.on_batch_folder)
        menu.addAction(act_folder)

        act_export = QAction("导出历史记录(&E)...", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self.on_export_history)
        menu.addAction(act_export)

        menu.addSeparator()
        act_quit = QAction("退出(&Q)", self)
        act_quit.triggered.connect(self.close)
        menu.addAction(act_quit)

        # ---------- 视图：主题切换 ----------
        view = self.menuBar().addMenu("视图(&V)")
        group = QActionGroup(self)
        group.setExclusive(True)
        self._theme_actions: dict[str, QAction] = {}
        for idx, key in enumerate(("light", "dark"), start=1):
            act = QAction(f"{THEME_LABELS[key]}主题(&{THEME_LABELS[key][0]})", self)
            act.setCheckable(True)
            act.setChecked(key == self._theme)
            act.setShortcut(f"Ctrl+{idx}")
            act.triggered.connect(lambda _checked=False, k=key: self.on_theme_changed(k))
            group.addAction(act)
            view.addAction(act)
            self._theme_actions[key] = act

        view.addSeparator()
        act_fit = QAction("适应窗口显示图片(&F)", self)
        act_fit.setShortcut("Ctrl+0")
        act_fit.triggered.connect(self._redraw_current_image)
        view.addAction(act_fit)

    def on_theme_changed(self, name: str) -> None:
        """切换外观主题，并写回配置（下次启动沿用）。"""
        self._theme = set_theme(name)
        self.setStyleSheet(build_qss())
        self._apply_card_shadows()
        # 徽章的半透明胶囊底色取自主题令牌、车牌展示卡则是内联样式，
        # 切主题后重放一次，避免留下上一个主题的残色
        if self._badge_state is not None:
            self._apply_badge(*self._badge_state)
        if self._history_placeholder:
            self._show_history_placeholder()
        self.config.ui.theme = self._theme
        for key, act in self._theme_actions.items():
            act.setChecked(key == self._theme)
        self._refresh_statusbar()
        logger.info("切换外观主题 → %s", THEME_LABELS.get(self._theme, self._theme))

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
        tool_card = QWidget()
        tool_card.setObjectName("toolCard")
        toolbar = QHBoxLayout(tool_card)
        toolbar.setContentsMargins(10, 8, 10, 8)
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
        # 用 objectName 而不是内联色：切主题时内联样式不会跟着变
        camera_caption.setObjectName("hintLabel")
        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_batch)
        toolbar.addWidget(self.btn_camera)
        toolbar.addSpacing(6)
        toolbar.addWidget(camera_caption)
        toolbar.addWidget(self.camera_combo)
        toolbar.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(160)
        self.progress.setVisible(False)
        self.progress_text = QLabel("")
        self.progress_text.setObjectName("statusChip")
        self.progress_text.setVisible(False)
        toolbar.addWidget(self.progress)
        toolbar.addWidget(self.progress_text)
        left_layout.addWidget(tool_card)

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

        # 号码 + 一键复制：识别之后最常见的下一步就是"把号码粘到别处去"
        plate_row = QWidget()
        plate_row_layout = QHBoxLayout(plate_row)
        plate_row_layout.setContentsMargins(0, 0, 0, 0)
        plate_row_layout.setSpacing(6)
        plate_row_layout.addWidget(self.plate_label, 1)
        self.btn_copy = QPushButton("复制")
        self.btn_copy.setObjectName("mini")
        self.btn_copy.setFixedWidth(54)
        self.btn_copy.setToolTip("复制当前号码到剪贴板（Ctrl+Shift+C）")
        self.btn_copy.setShortcut("Ctrl+Shift+C")
        self.btn_copy.clicked.connect(self.on_copy_text)
        self.btn_copy.setEnabled(False)
        plate_row_layout.addWidget(self.btn_copy, 0, Qt.AlignBottom)

        self.roi_label = QLabel()
        self.roi_label.setObjectName("roiArea")
        self.roi_label.setAlignment(Qt.AlignCenter)
        self.roi_label.setMinimumSize(240, 88)

        self.meta_label = QLabel("等待识别…")
        self.meta_label.setWordWrap(True)
        self.meta_label.setObjectName("metaLabel")

        # 多车牌场景下这里列出全部候选（结果卡位只放最高分的那个，
        # 此前其余的只写进日志，界面上看不见）
        self.candidates_table = QTableWidget(0, 3)
        self.candidates_table.setHorizontalHeaderLabels(["号码", "颜色", "置信度"])
        self.candidates_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.candidates_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.candidates_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.candidates_table.setAlternatingRowColors(True)
        self.candidates_table.setShowGrid(False)
        self.candidates_table.verticalHeader().setVisible(False)
        self.candidates_table.verticalHeader().setDefaultSectionSize(26)
        self.candidates_table.setMaximumHeight(104)
        self.candidates_table.setVisible(False)
        self.candidates_table.setToolTip("同一张图检出的全部号码，按置信度排序；点行可回显")
        self.candidates_table.cellClicked.connect(self._on_candidate_clicked)

        result_layout.addWidget(plate_row, 0, 0, 1, 2)
        result_layout.addWidget(self.roi_label, 1, 0)
        color_card = QLabel("—")
        color_card.setObjectName("colorBadge")
        color_card.setAlignment(Qt.AlignCenter)
        self.color_card = color_card
        result_layout.addWidget(color_card, 1, 1)
        result_layout.addWidget(self.meta_label, 2, 0, 1, 2)
        result_layout.addWidget(self.candidates_table, 3, 0, 1, 2)
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
        # 点击历史行 → 把该条结果回显到识别结果区（号码皮肤/颜色徽章/ROI/元信息）
        self.history_table.cellClicked.connect(self._on_history_row_clicked)
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

        # 日志文件的去处：同一天多次启动写同一个文件，用户报障时这是唯一的证据。
        # 路径可选中复制（有些机器上文件管理器打不开，那时只能靠手动去找）。
        log_row = QHBoxLayout()
        self.log_path_label = QLabel()
        self.log_path_label.setObjectName("hintLabel")
        self.log_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log_path_label.setToolTip("本次运行的日志文件；同一天多次启动会追加到同一个文件")
        self.btn_log_dir = QPushButton("打开日志目录")
        self.btn_log_dir.setObjectName("secondary")
        self.btn_log_dir.clicked.connect(self.on_open_log_dir)
        log_row.addWidget(self.log_path_label, 1)
        log_row.addWidget(self.btn_log_dir)
        log_layout.addLayout(log_row)
        right_layout.addWidget(log_box)
        self._refresh_log_path()

        splitter.addWidget(right)
        splitter.setSizes([720, 480])

        # 卡片引用，供统一加投影（层次感）
        self._cards = [engine_box, result_box, history_box, log_box, tool_card]
        self._show_history_placeholder()

    # ---------------------------------------------------------------- 状态栏 / 装饰

    def _build_statusbar(self) -> None:
        """底部状态栏：一眼看到版本、当前引擎、隐私策略与设备情况。"""
        bar = self.statusBar()
        self._status_engine = QLabel()
        self._status_privacy = QLabel()
        self._status_camera = QLabel()
        for label in (self._status_engine, self._status_privacy, self._status_camera):
            label.setObjectName("statusChip")
            bar.addPermanentWidget(label)
        bar.addPermanentWidget(QLabel(f"v{__version__}"))
        self._refresh_statusbar()

    def _refresh_statusbar(self) -> None:
        # 引擎以下拉框实际选中项为准（与界面所见一致，而不是配置文件里
        # 可能已失效的默认值）
        name = self.engine_combo.currentData() or self.config.engine.default
        self._status_engine.setText(f"引擎：{name}")
        self._status_privacy.setText(
            "隐私：输出打码" if self.privacy.mask_output else "隐私：未打码"
        )
        self._status_camera.setText("")

    def _apply_card_shadows(self) -> None:
        """给卡片加一点柔和投影（Qt 无 CSS box-shadow，用图形效果实现）。

        幂等：切主题时复用已有 effect 只改颜色，否则每切一次都会叠一层新
        效果对象（旧的不回收，而且叠加后投影会越来越黑）。
        """
        r, g, b, a = (int(part) for part in _T["shadow"].split(","))
        for widget in self._cards:
            effect = widget.graphicsEffect()
            if not isinstance(effect, QGraphicsDropShadowEffect):
                effect = QGraphicsDropShadowEffect(widget)
                widget.setGraphicsEffect(effect)
            effect.setBlurRadius(18)
            effect.setOffset(0, 2)
            effect.setColor(QColor(r, g, b, a))

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
        # 配置的默认引擎不可用时会回落到第一项——把配置同步成实际选中的，
        # 否则界面跑的引擎和 config.engine.default 各说各话（状态栏曾因此
        # 显示 cascade、实际跑的是 cv）
        selected = self.engine_combo.itemData(default_idx)
        if selected != self.config.engine.default:
            logger.info(
                "配置的默认引擎 %s 不可用，界面实际使用 %s", self.config.engine.default, selected
            )
            self.config.engine.default = selected
        self._update_engine_hint()
        self._refresh_statusbar()

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
        self._refresh_statusbar()
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
        self.progress_text.setVisible(True)
        self._update_progress_text()
        self._next_in_queue()

    def _update_progress_text(self) -> None:
        self.progress_text.setText(f"{self._batch_done} / {self._batch_total}")

    def _next_in_queue(self) -> None:
        if not self._batch_queue:
            self.progress.setVisible(False)
            self.progress_text.setVisible(False)
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
            self._update_progress_text()
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
        self._update_progress_text()
        self._worker = None
        QTimer.singleShot(30, self._next_in_queue)  # 让 UI 呼吸一下

    def _on_image_failed(self, source: str, error: str) -> None:
        logger.error("识别失败 %s: %s", source, error)
        QMessageBox.critical(self, "识别失败", f"{source}\n\n{error}")
        self._batch_done += 1
        self.progress.setValue(self._batch_done)
        self._update_progress_text()
        self._worker = None
        QTimer.singleShot(30, self._next_in_queue)

    def _apply_badge(self, key: str, label: str | None = None, muted: bool = False) -> None:
        """颜色徽章：浅底胶囊。Qt QSS 不支持 #RRGGBBAA，半透明只能用 rgba()。"""
        # 记下状态：切主题时要按新令牌重放一次胶囊底色
        self._badge_state = (key, label, muted)
        if muted or label is None:
            self.color_card.setText("—" if muted else (label or ""))
            self.color_card.setStyleSheet(
                f"background:rgba(138, 147, 163, 0.10);"
                f"border:1px solid rgba(138, 147, 163, 0.40); color:{_T['muted']};"
            )
            return
        self.color_card.setText(label)
        hex_color = _BADGE_HEX.get(key, _T["muted"])
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
        self.color_card.setStyleSheet(
            f"background:rgba({r}, {g}, {b}, 0.14);"
            f"border:1px solid rgba({r}, {g}, {b}, 0.55);"
            f"color:{hex_color};"
        )

    def _show_image(self, img: np.ndarray) -> None:
        w = max(320, self.image_label.width() - 8)
        h = max(240, self.image_label.height() - 8)
        self.image_label.setPixmap(_array_to_pixmap(img, w, h))
        self._current_image = img

    def _redraw_current_image(self) -> None:
        """按当前容器尺寸重绘静态图片。

        ``QLabel#imageArea`` 刻意没有开 ``scaledContents``：让它自己拉伸会把
        位图放大到模糊（`_array_to_pixmap` 的平滑缩放才有正确的质量）。代价是
        缩放窗口后 pixmap 不会自己跟着变，所以这里补一次重绘。
        """
        img = self._current_image
        if img is not None:
            self._show_image(img)

    def resizeEvent(self, event) -> None:
        """窗口尺寸变化 → 去抖后重绘图片。

        去抖 200ms：拖动边框时每个像素变化都重采样一遍会明显卡顿。
        """
        super().resizeEvent(event)
        self._fit_timer.start(200)

    # ---------------------------------------------------------------- 复制

    def _set_current_text(self, text: str) -> None:
        """记录当前号码，并同步「复制」按钮的可用性。"""
        self._current_text = text
        self.btn_copy.setEnabled(bool(text))

    def on_copy_text(self) -> None:
        """把当前展示的号码复制到剪贴板。"""
        text = self._current_text
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"已复制：{text}", 2000)

    def _fill_candidates(self, results: list[PlateResult]) -> None:
        """把全部候选填进候选表；只有一个候选时整张表隐藏（不占地方）。"""
        self.candidates_table.setRowCount(0)
        if len(results) <= 1:
            self.candidates_table.setVisible(False)
            return

        self.candidates_table.setVisible(True)
        for res in results:
            row = self.candidates_table.rowCount()
            self.candidates_table.insertRow(row)
            values = [res.text or "（未读出）", res.color.label, f"{res.confidence:.2f}"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                # 与历史表同口径：结果对象绑在单元格上，点行即可回显
                item.setData(Qt.UserRole, res)
                if col == 0:
                    item.setForeground(QColor(_BADGE_HEX.get(res.color.value, _T["text"])))
                else:
                    item.setTextAlignment(Qt.AlignCenter)
                self.candidates_table.setItem(row, col, item)

    def _on_candidate_clicked(self, row: int, _col: int) -> None:
        """点击候选行 → 把该候选回显到结果卡位（不改写历史）。"""
        item = self.candidates_table.item(row, 0)
        if item is None:
            return
        res = item.data(Qt.UserRole)
        if isinstance(res, PlateResult):
            self._render_result(res, prefix="候选")

    def _show_results(self, results: list[PlateResult], elapsed: float, source: str) -> None:
        if not results:
            self._apply_plate_skin(PlateColor.UNKNOWN.value, "未识别到车牌")
            self._apply_badge(PlateColor.UNKNOWN.value, muted=True)
            self.meta_label.setText(f"耗时 {elapsed:.0f} ms\n{source}")
            self._set_current_text("")
            self._fill_candidates([])
            return

        best = results[0]
        self._render_result(best, prefix=f"耗时 {elapsed:.0f} ms")
        self._fill_candidates(results)

        if len(results) > 1:
            logger.info("共 %d 个车牌: %s", len(results), " / ".join(r.text for r in results))

        self._append_history(best)

    def _render_result(self, res: PlateResult, prefix: str = "") -> None:
        """把一条结果渲染到结果卡位（号码皮肤 / 颜色徽章 / ROI / 元信息）。

        结果区、历史行点击、候选行点击三处共用这一段 —— 此前历史行点击
        与实时结果各写了一份几乎相同的渲染代码，改一处漏一处。
        """
        self._apply_plate_skin(res.color.value, res.text)
        self._apply_badge(res.color.value, label=res.color.label)
        self._set_current_text(res.text)

        if res.roi is not None and res.roi.size:
            self._last_roi_pixmap = _array_to_pixmap(
                res.roi, self.roi_label.width(), self.roi_label.height()
            )
            self.roi_label.setPixmap(self._last_roi_pixmap)

        chars_info = (
            " ".join(f"{t}:{c:.2f}" for t, c in zip(res.text, res.char_confidences, strict=False))
            if res.char_confidences
            else ""
        )
        head = f"{prefix}｜" if prefix else ""
        self.meta_label.setText(
            f"{head}引擎 {res.engine}｜置信度 {res.confidence:.2f}\nbbox={res.bbox}  {chars_info}"
        )

    def _apply_plate_skin(self, key: str, text: str) -> None:
        """把号码渲染成「车牌的样子」：按国标配色铺底、对比色写字。

        比单纯给文字染色信息量大得多 —— 一眼能看出蓝牌/黄牌/新能源绿牌。
        """
        skin = _PLATE_SKIN.get(key, _PLATE_SKIN[PlateColor.UNKNOWN.value])
        self.plate_label.setText(text)
        self.plate_label.setStyleSheet(
            f"background:{skin['bg']}; color:{skin['fg']};"
            f"border:2px solid {skin['border']}; border-radius:12px;"
        )

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
            self._status_camera.setText("设备：无摄像头")
            return
        for idx in found:
            self.camera_combo.addItem(f"摄像头 {idx}", idx)
        self.camera_combo.setEnabled(True)
        self._status_camera.setText(f"设备：{len(found)} 个摄像头")

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
        # 同时清掉缓存的图，否则窗口一缩放它又会被重绘回来
        self._current_image = None
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
        self._fit_timer.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        if self._camera is not None:
            self._camera.stop()
            self._camera.wait(2000)
        super().closeEvent(event)

    # ---------------------------------------------------------------- 拖放

    def _dropped_images(self, urls) -> list[Path]:
        """把拖进来的 URL 展开成待识别图片列表（文件夹按扩展名白名单展开）。"""
        files: list[Path] = []
        for url in urls:
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir():
                files.extend(
                    sorted(
                        p
                        for p in path.iterdir()
                        if p.is_file() and p.suffix.lower() in self.limits.allowed_extensions
                    )
                )
            elif path.is_file() and path.suffix.lower() in self.limits.allowed_extensions:
                files.append(path)
        return files

    def dragEnterEvent(self, event) -> None:
        """接受拖入（含文件夹）。判据是"有本地路径"，不预先遍历目录 ——
        dragEnter 在鼠标移动过程中会被高频调用，这里做 I/O 会卡住 UI。"""
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        files = self._dropped_images(event.mimeData().urls())
        if not files:
            QMessageBox.information(
                self, "提示", "拖入的内容里没有可识别的图片文件（支持常见图片扩展名）。"
            )
            return
        event.acceptProposedAction()
        logger.info("拖放导入 %d 张图片", len(files))
        self._start_batch(files)

    # ---------------------------------------------------------------- 历史

    def _show_history_placeholder(self) -> None:
        """空表格太冷清，放一行居中的灰色提示（首次识别时自动移除）。"""
        self.history_table.clearSpans()
        self.history_table.setRowCount(1)
        item = QTableWidgetItem("暂无记录 — 识别后会自动出现在这里")
        item.setTextAlignment(Qt.AlignCenter)
        item.setForeground(QColor(_T["muted"]))
        self.history_table.setItem(0, 0, item)
        self.history_table.setSpan(0, 0, 1, self.history_table.columnCount())
        self._history_placeholder = True

    def _drop_history_placeholder(self) -> None:
        if not self._history_placeholder:
            return
        self.history_table.clearSpans()
        self.history_table.setRowCount(0)
        self._history_placeholder = False

    def _append_history(self, res: PlateResult) -> None:
        self._drop_history_placeholder()
        self._history.append(res)
        if len(self._history) > self.config.ui.max_history:
            self._history = self._history[-self.config.ui.max_history :]
            # 列表被截断时表格也要同步删首行，否则行号与 _history 下标错位
            self.history_table.removeRow(0)

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
            # 把结果对象绑到该行的每个单元格上，点击任意列都能取回
            item.setData(Qt.UserRole, res)
            if col == 1:
                font = QFont()
                font.setBold(True)
                item.setFont(font)
                item.setForeground(QColor(_BADGE_HEX.get(res.color.value, _T["text"])))
            if col in (2, 3, 4):
                item.setTextAlignment(Qt.AlignCenter)
            self.history_table.setItem(row, col, item)
        self.history_table.scrollToBottom()

    def _on_history_row_clicked(self, row: int, _col: int) -> None:
        """点击历史行：把该条识别结果回显到上方结果区。

        只更新展示，不再写历史（否则会自己追加自己）。
        """
        item = self.history_table.item(row, 0)
        if item is None:
            return
        res = item.data(Qt.UserRole)
        if isinstance(res, PlateResult):
            self._render_result(res, prefix="历史记录")

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
        self._history_placeholder = False
        self._show_history_placeholder()

    # ---------------------------------------------------------------- 日志

    #: 「运行日志」面板最多显示多少行（环形缓冲容量是 500，这里只截最近的）
    _LOG_VIEW_LINES = 200

    def _refresh_log_path(self) -> None:
        """把日志文件路径显示在面板下方（未开启文件日志时如实说明）。"""
        path = current_log_file()
        if path is None:
            self.log_path_label.setText("日志文件：未开启（config.json 的 log_to_file）")
            self.btn_log_dir.setEnabled(False)
            return
        self.btn_log_dir.setEnabled(True)
        self.log_path_label.setText(f"日志文件：{path.absolute()}")

    def _refresh_log_view(self) -> None:
        """把内存缓冲里最近的日志刷到面板。

        刷新判据用**内容快照**而不是行数：缓冲满了（500 条）之后行数恒定，
        早期的"行数相同就返回"会让面板在跑起来之后彻底冻住 —— 这正是
        v2.7.2 之前界面显示不出日志的第二个原因。
        """
        records = memory_handler.dump()[-self._LOG_VIEW_LINES :]
        if records == self._log_snapshot:
            return
        self._log_snapshot = records
        self.log_view.setRowCount(0)
        for line in records:
            row = self.log_view.rowCount()
            self.log_view.insertRow(row)
            self.log_view.setItem(row, 0, QTableWidgetItem(line))
        self.log_view.scrollToBottom()
        self._refresh_log_path()

    def on_open_log_dir(self) -> None:
        """在系统文件管理器里打开日志所在目录。"""
        path = current_log_file()
        if path is None:
            QMessageBox.information(
                self,
                "运行日志",
                "本次运行没有写日志文件。\n\n"
                '把 config.json 里的 "log_to_file" 设为 true 后重启即可开启；\n'
                '日志会写到 "log_dir"（默认 logs/）目录下。',
            )
            return
        error = _open_directory(path.parent)
        if error:
            QMessageBox.information(
                self,
                "运行日志",
                f"打不开文件管理器（{error}）。\n\n日志文件在这里：\n{path.absolute()}",
            )


def run_ui(config: Config | None = None) -> None:
    """启动界面（独立入口）。"""
    config = config or Config.load()
    # 界面此前**从未**初始化过日志：``setup_logging`` 只在 CLI 里调过，
    # 于是「运行日志」面板永远空白，而且 ``logger.propagate`` 一直是 True
    # （日志顺着根 logger 走，INFO 级被直接丢掉）。这里补上，幂等。
    setup_logging(**config.logging_kwargs())

    app = QApplication(sys.argv)
    app.setApplicationName("车牌识别系统 v2")

    win = MainWindow(config)
    win.show()
    sys.exit(app.exec())


__all__ = ["MainWindow", "run_ui"]
