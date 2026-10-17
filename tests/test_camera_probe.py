"""摄像头探测与错误提示的回归测试。

背景（v2.3.7）：用户在**没有插入任何摄像头**的机器上点「启动摄像头」，
弹窗只有一句「无法打开摄像头 0」——分不清是没插线、被占用还是索引不对，
弹完画面永远停在「等待识别」。改进后：

1. 打不开时先扫一遍系统里有哪些可用索引；
2. 一个都没有 → 给出可行动的排查清单（插线 / 设备管理器 / 独占 / 隐私开关）；
3. 有但不是当前索引 → 直接列出可用索引，并指向工具栏新加的下拉框。

探测逻辑 ``list_cameras`` 与提示构造 ``camera_error_message`` 都是纯函数，
用假的开设备对象即可测试，不需要真摄像头。
"""

from __future__ import annotations

from lpr.ui.worker import camera_error_message, list_cameras


class FakeCap:
    """可控的 ``cv2.VideoCapture`` 替身。

    ``openable`` 决定 ``isOpened()``；``frame_ok`` 决定 ``grab()``——
    部分虚拟设备能通过 ``isOpened`` 却出不了帧，探测必须两者都查。
    """

    def __init__(self, openable: bool = True, frame_ok: bool = True):
        self.openable = openable
        self.frame_ok = frame_ok
        self.released = False

    def isOpened(self) -> bool:
        return self.openable

    def grab(self) -> bool:
        return self.frame_ok

    def release(self) -> None:
        self.released = True


class TestListCameras:
    def test_returns_indices_that_open_and_grab(self):
        def opener(idx: int):
            return FakeCap(openable=(idx in (0, 2)))

        assert list_cameras(4, open_camera=opener) == [0, 2]

    def test_skips_indices_that_open_but_never_produce_a_frame(self):
        """isOpened 为真但 grab 失败 = 幽灵/虚拟设备，不算可用。"""

        def opener(_idx: int):
            return FakeCap(openable=True, frame_ok=False)

        assert list_cameras(3, open_camera=opener) == []

    def test_releases_every_probed_device(self):
        """探测完必须释放——探测脚本把摄像头句柄攥着不放是常见事故。"""
        caps: list[FakeCap] = []

        def opener(_idx: int):
            cap = FakeCap(openable=False)
            caps.append(cap)
            return cap

        list_cameras(3, open_camera=opener)
        assert len(caps) == 3
        assert all(cap.released for cap in caps)

    def test_a_raising_device_does_not_abort_the_scan(self):
        """某个设备驱动抛异常时跳过它，其余照常探测。"""
        good = FakeCap()

        def opener(idx: int):
            if idx == 1:
                raise OSError("driver boom")
            return good if idx == 0 else FakeCap(openable=False)

        assert list_cameras(3, open_camera=opener) == [0]

    def test_no_devices_returns_empty_list(self):
        assert list_cameras(2, open_camera=lambda _i: FakeCap(openable=False)) == []

    def test_concurrent_probes_never_overlap(self):
        """探测必须串行 —— 这是防进程级崩溃的护栏，不是性能优化。

        Windows 的 DirectShow/MSMF 后端并发 ``VideoCapture`` + ``release``
        会触发 ``0xc0000374``（堆损坏，Python 层抓不住）。界面给每个窗口起
        一个探测线程，测试里一个进程连建十几个窗口，所以必须锁住。

        判据：两个线程同时进来，任一时刻"正在探测"的计数不得大于 1。
        """
        import threading
        import time

        state = {"inside": 0, "peak": 0}
        lock = threading.Lock()

        def opener(_idx: int):
            with lock:
                state["inside"] += 1
                state["peak"] = max(state["peak"], state["inside"])
            time.sleep(0.01)
            with lock:
                state["inside"] -= 1
            return FakeCap(openable=False)

        threads = [
            threading.Thread(target=lambda: list_cameras(3, open_camera=opener)) for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state["peak"] == 1


class TestCameraErrorMessage:
    def test_no_devices_gives_an_actionable_checklist(self):
        msg = camera_error_message(0, [])
        # 排查项必须覆盖最常见的四个原因
        assert "设备管理器" in msg
        assert "独占" in msg
        assert "隐私" in msg
        # 不应该再出现只有索引号的旧句子
        assert msg.startswith("未检测到")

    def test_available_devices_are_listed_with_indices(self):
        msg = camera_error_message(0, [1, 2])
        assert "1、2" in msg
        assert "摄像头 0" in msg  # 说清楚失败的是哪个
        assert "下拉框" in msg  # 告诉用户入口在哪

    def test_message_for_nonzero_index(self):
        msg = camera_error_message(3, [0])
        assert "摄像头 3" in msg
        assert "0" in msg
