"""引擎注册表：按名字查找/创建引擎实例。

这是"双引擎可切换"的关键：界面层只需要 ``get_engine(cfg.engine.default)``，
新增第三种引擎时不用改动任何调用方代码。
"""

from __future__ import annotations

from typing import Any

from ..config import EngineConfig
from .base import BaseEngine

#: 引擎名 → 引擎类
_REGISTRY: dict[str, type[BaseEngine]] = {}
#: 创建过的单例（引擎加载模型开销大，应复用）
_INSTANCES: dict[str, BaseEngine] = {}


def register_engine(cls: type[BaseEngine]) -> type[BaseEngine]:
    """注册引擎类，可作为装饰器使用。"""
    if not cls.name or cls.name == "base":
        raise ValueError(f"{cls.__name__} 必须定义唯一的 name")
    _REGISTRY[cls.name] = cls
    return cls


def available_engines() -> list[str]:
    """返回所有已注册的引擎名。"""
    return sorted(_REGISTRY)


def describe_engines() -> list[dict[str, Any]]:
    """返回引擎清单及可用性，供界面渲染下拉框。"""
    out = []
    for name in available_engines():
        cls = _REGISTRY[name]
        ok, reason = cls.is_available()
        out.append(
            {
                "name": name,
                "display_name": cls.display_name,
                "description": cls.description,
                "available": ok,
                "reason": reason,
            }
        )
    return out


def create_engine(name: str, config: EngineConfig | None = None, **kwargs: Any) -> BaseEngine:
    """按名字新建引擎实例（不缓存）。"""
    if name not in _REGISTRY:
        raise KeyError(
            f"未知引擎 {name!r}，可用: {', '.join(available_engines()) or '（无）'}"
        )
    cls = _REGISTRY[name]
    # CVEngine / DeepEngine 都接受 config 关键字；其他引擎用 **kwargs 兜底
    try:
        return cls(config=config, **kwargs)
    except TypeError:
        return cls(**kwargs)


def get_engine(
    name: str,
    config: EngineConfig | None = None,
    *,
    preload: bool = True,
    **kwargs: Any,
) -> BaseEngine:
    """取得引擎单例并按需预加载模型。

    Args:
        name: 引擎名，见 ``available_engines()``。
        config: 引擎配置。
        preload: 是否立即加载模型（默认 True，把耗时从首次识别挪到启动时）。
    """
    if name not in _INSTANCES:
        _INSTANCES[name] = create_engine(name, config, **kwargs)
    engine = _INSTANCES[name]
    if preload:
        engine.ensure_ready()
    return engine


def clear_instances() -> None:
    """清空单例缓存（测试用）。"""
    for engine in _INSTANCES.values():
        try:
            engine.close()
        except Exception:
            pass
    _INSTANCES.clear()


__all__ = [
    "register_engine",
    "available_engines",
    "describe_engines",
    "create_engine",
    "get_engine",
    "clear_instances",
]
