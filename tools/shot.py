"""界面截图工具：offscreen 渲染主窗口到 PNG，供美化前后对比与人工检查。

用法::

    python tools/shot.py out.png [--sample samples/1.jpg]

用真实样张跑一次识别，填上画面/结果/历史，产出接近真实使用状态的截图。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# offscreen 平台默认找不到系统中文字体（渲染成豆腐块），显式指定字体目录
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtWidgets import QApplication

from lpr.config import Config
from lpr.ui.window import MainWindow


def _pump(app: QApplication, seconds: float = 0.4) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "shot.png")
    sample = None
    if "--sample" in sys.argv:
        sample = Path(sys.argv[sys.argv.index("--sample") + 1])

    app = QApplication([])
    win = MainWindow(Config.load())
    win.resize(1240, 820)
    win.show()
    _pump(app, 0.5)  # 等摄像头探测线程与信号投递

    if sample is not None and sample.exists():
        from lpr.engines import get_engine
        from lpr.utils.image import imread

        img = imread(sample)
        win._show_image(img)
        results = get_engine("deep", win.config.engine).predict(img)
        win._show_results(results, 137.0, str(sample))
        # 伪造几条历史，让表格有真实感（仅截图工具使用，不影响软件行为）
        for res in results[:3]:
            win._append_history(res)
        _pump(app, 0.3)

    for _ in range(20):
        app.processEvents()

    win.grab().save(str(out))
    print("saved", out, out.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
