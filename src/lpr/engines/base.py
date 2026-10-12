"""识别引擎统一接口。

界面层 / CLI 层只依赖 ``BaseEngine``，不关心底层是 SVM 还是神经网络。
新增引擎只需继承本类并注册到 ``engines.registry``。
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np

from ..types import PlateResult


class EngineUnavailable(RuntimeError):
    """引擎依赖缺失或模型加载失败，无法使用。

    与"识别失败"不同：识别失败返回空列表，依赖缺失抛本异常，
    这样界面可以给出"请安装 XXX"的明确提示，而不是静默无结果。
    """


class BaseEngine(abc.ABC):
    """所有识别引擎的基类。"""

    #: 引擎唯一标识，用于配置与注册表，例如 ``"cv"``
    name: str = "base"
    #: 展示名称
    display_name: str = "基础引擎"
    #: 简要说明
    description: str = ""

    def __init__(self, **kwargs: object) -> None:
        self._ready = False

    # ------------------------------------------------------------------ 生命周期

    @abc.abstractmethod
    def load(self) -> None:
        """加载模型/建立会话。耗时操作应放在这里，而不是每次识别时。"""

    def close(self) -> None:  # noqa: B027 —— 有意做成可选钩子，子类按需覆写
        """释放资源，默认无操作。"""

    @property
    def ready(self) -> bool:
        return self._ready

    def ensure_ready(self) -> None:
        if not self._ready:
            self.load()

    # ------------------------------------------------------------------ 可用性探测

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """返回（是否可用, 原因说明）。

        界面用它在引擎下拉框里禁用不可用的引擎，并给出提示。
        """
        return True, ""

    # ------------------------------------------------------------------ 核心接口

    @abc.abstractmethod
    def predict(self, image: np.ndarray | str | Path) -> list[PlateResult]:
        """识别一张图中的所有车牌。

        Args:
            image: BGR 图像数组，或图片路径。

        Returns:
            识别结果列表；没找到车牌时返回空列表（不要返回 ``None``）。
        """

    def predict_path(self, path: str | Path) -> list[PlateResult]:
        """便捷方法：直接传路径。"""
        return self.predict(path)

    # ------------------------------------------------------------------ 杂项

    def __enter__(self) -> BaseEngine:
        self.ensure_ready()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "ready" if self._ready else "unloaded"
        return f"<{type(self).__name__} name={self.name!r} {state}>"


__all__ = ["BaseEngine", "EngineUnavailable"]
