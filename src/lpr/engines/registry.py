"""引擎注册表：按名字查找/创建引擎实例。

这是"双引擎可切换"的关键：界面层只需要 ``get_engine(cfg.engine.default)``，
新增第三种引擎时不用改动任何调用方代码。
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from typing import Any

from ..config import EngineConfig
from .base import BaseEngine

#: 推理串行锁。
#:
#: 原因：*引擎实例是跨线程共享的单例*（见 ``get_engine``），而底层推理后端
#: （ONNX Runtime / OpenCV）复用同一份会话/模型时并发调用并不安全，且多线程
#: 同时推理只会互相抢 CPU（Python 侧无 GIL 释放的收益）。
#: 因此把所有 ``predict`` 调用串行化：批量识别与摄像头识别不会同时跑模型。
#:
#: 注意：多**进程**批量识别（``Recognizer(workers=N)``）不受影响 ——
#: 各进程有各自的解释器与锁实例。
INFERENCE_LOCK = threading.RLock()

#: 引擎名 → 引擎类
_REGISTRY: dict[str, type[BaseEngine]] = {}
#: 创建过的单例（引擎加载模型开销大，应复用）
_INSTANCES: dict[str, BaseEngine] = {}
#: 单例对应的配置指纹，配置变化时重建（见 ``get_engine``）
_INSTANCE_KEYS: dict[str, str] = {}


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


def _config_key(config: EngineConfig | None) -> str:
    """把引擎配置压成一个指纹，用于判断单例是否还能复用。"""
    if config is None:
        return "default"
    return json.dumps(asdict(config), sort_keys=True, ensure_ascii=False, default=str)


def get_engine(
    name: str,
    config: EngineConfig | None = None,
    *,
    preload: bool = True,
    **kwargs: Any,
) -> BaseEngine:
    """取得引擎单例并按需预加载模型。

    单例按 ``(name, 配置指纹)`` 缓存：**传入不同配置时会自动重建**。
    （v2.3.1 修复：此前只按 name 缓存，第二次传入的新配置被静默忽略，
    必须手动调 ``clear_instances()`` 才生效。）

    Args:
        name: 引擎名，见 ``available_engines()``。
        config: 引擎配置。
        preload: 是否立即加载模型（默认 True，把耗时从首次识别挪到启动时）。
    """
    key = _config_key(config)
    if name not in _INSTANCES or _INSTANCE_KEYS.get(name) != key:
        if name in _INSTANCES:
            # 配置变了：先关掉旧实例再重建，避免模型/句柄泄漏
            try:
                _INSTANCES[name].close()
            except Exception:  # noqa: BLE001 - 释放失败不应阻断重建
                pass
        _INSTANCES[name] = create_engine(name, config, **kwargs)
        _INSTANCE_KEYS[name] = key

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
    _INSTANCE_KEYS.clear()


__all__ = [
    "register_engine",
    "available_engines",
    "describe_engines",
    "create_engine",
    "get_engine",
    "clear_instances",
    "INFERENCE_LOCK",
]
