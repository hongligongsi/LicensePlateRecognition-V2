"""引擎注册表：按名字查找/创建引擎实例。

这是"双引擎可切换"的关键：界面层只需要 ``get_engine(cfg.engine.default)``，
新增第三种引擎时不用改动任何调用方代码。
"""

from __future__ import annotations

import contextlib
import json
import threading
from dataclasses import asdict
from typing import Any

from ..config import CASCADE_RESOLVE_ORDER, EngineConfig
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


#: **策略别名**：能从 ``--engine`` 与 ``config.engine.default`` 传进来，但
#: **不是**注册表里的具体引擎。
#:
#: 为什么不给它注册一个同名引擎类：
#:
#: * ``cascade`` 不是一类实现，而是"用哪个引擎"的规则 —— ``get_engine("cascade")``
#:   本来就无处可查，解析发生在 ``Recognizer._resolve_engine``；
#: * ``FusionEngine`` 的成员过滤（``NON_MEMBER_NAMES`` 里含 ``cascade`` / ``auto``）
#:   正依赖"``cascade`` 不是引擎名"这个事实，真注册一个同名引擎会让别名被当成
#:   融合成员。
#:
#: 但它对使用者是**合法取值**（``config.ENGINE_CHOICES`` 里就有），所以界面下拉框
#: 必须能选中它。此前下拉框只列 ``describe_engines()``（纯真引擎），配置里的
#: ``cascade`` 找不到对应项，被当成"不可用"静默降级成第一项 ``cv`` ——
#: 实测代价：11 张人工真值样张的命中从 10/11 掉到 0/11。
ALIAS_ENGINES: dict[str, dict[str, Any]] = {
    "cascade": {
        "display_name": "级联引擎",
        "description": "CV 快速定位 → 深度精确识别（等同 deep 引擎）",
        # 引用 config 的同一常量：界面说"cascade 会退化成谁"与运行时真的
        # 退化成谁，必须是同一份知识
        "resolve_order": CASCADE_RESOLVE_ORDER,
        # 别名在下拉框里紧跟它实际使用的引擎（deep），而不是按字母序落到最后
        "anchor": CASCADE_RESOLVE_ORDER[0],
    },
}


def describe_engine_choices() -> list[dict[str, Any]]:
    """引擎清单**含策略别名**，供界面下拉框与 CLI 引擎表渲染。

    与 :func:`describe_engines` 的唯一区别：把 ``cascade`` 这类别名也列出来，
    并排在它实际使用的引擎之后。别名的可用性取自它解析出的引擎 ——
    ``cascade`` 在 ``deep`` 不可用时会退化成纯 ``cv``，此时仍算可用，
    但 ``description`` 会写明退化，免得界面提示与实际跑的引擎不一致。

    真引擎那一段与 :func:`describe_engines` 完全同源，不做任何改写。
    """
    rows = describe_engines()
    by_name = {row["name"]: row for row in rows}
    for alias, meta in ALIAS_ENGINES.items():
        order: tuple[str, ...] = meta["resolve_order"]
        picked = next((n for n in order if by_name.get(n, {}).get("available")), None)
        if picked is None:
            entry: dict[str, Any] = {
                "name": alias,
                "display_name": meta["display_name"],
                "description": meta["description"],
                "available": False,
                "reason": "、".join(
                    f"{n} 不可用（{by_name.get(n, {}).get('reason') or '未注册'}）" for n in order
                ),
                "strategy": True,
            }
        else:
            description = meta["description"]
            if picked != order[0]:
                # 退化要写在 description 里：界面提示读的就是这个字段，
                # 只说"可用"会让用户以为跑的还是 deep
                description = f"{description}；{order[0]} 不可用，本次退化为 {picked.upper()}"
            entry = {
                "name": alias,
                "display_name": meta["display_name"],
                "description": description,
                "available": True,
                "reason": "",
                "strategy": True,
            }
        anchor_name = meta.get("anchor", picked)
        anchor = by_name.get(anchor_name)
        index = rows.index(anchor) + 1 if anchor is not None else len(rows)
        rows.insert(index, entry)
    return rows


def create_engine(name: str, config: EngineConfig | None = None, **kwargs: Any) -> BaseEngine:
    """按名字新建引擎实例（不缓存）。"""
    if name not in _REGISTRY:
        raise KeyError(f"未知引擎 {name!r}，可用: {', '.join(available_engines()) or '（无）'}")
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
            with contextlib.suppress(Exception):
                _INSTANCES[name].close()
        _INSTANCES[name] = create_engine(name, config, **kwargs)
        _INSTANCE_KEYS[name] = key

    engine = _INSTANCES[name]
    if preload:
        engine.ensure_ready()
    return engine


def clear_instances() -> None:
    """清空单例缓存（测试用）。"""
    for engine in _INSTANCES.values():
        with contextlib.suppress(Exception):
            engine.close()
    _INSTANCES.clear()
    _INSTANCE_KEYS.clear()


__all__ = [
    "INFERENCE_LOCK",
    "available_engines",
    "clear_instances",
    "create_engine",
    "describe_engines",
    "get_engine",
    "register_engine",
]
