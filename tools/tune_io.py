"""调参结果落盘的共用工具。

两个调参脚本（``tune_thresholds`` 无监督 / ``eval_locate`` 有监督）都需要把
最优 ``LocateConfig`` 参数写进真正会被读取的配置文件。

**关键坑**：``lpr.config.Config.load()`` 只读项目根的 ``config.json``，
而且 ``locate`` 参数嵌在 ``engine.locate`` 下（不是顶层 ``locate``）。
早期版本把参数写到顶层的 ``config.tuned.json``，既不生效也层级错误。

本模块统一处理：
* 定位到 ``engine.locate`` 的正确嵌套位置；
* 首次写入前把原 ``config.json`` 备份成 ``config.json.bak``；
* 元组自动转 list（JSON 不支持 tuple）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

CONFIG_NAME = "config.json"
BACKUP_NAME = "config.json.bak"


def apply_locate_params(params: dict, root: Path) -> Path:
    """把 *params* 合并进 ``<root>/config.json`` 的 ``engine.locate``。

    Args:
        params: ``{参数名: 值}``，值可以是 tuple（会转成 list）。
        root: 项目根目录。

    Returns:
        实际写入的配置文件路径。
    """
    cfg_path = root / CONFIG_NAME
    existing: dict = {}
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        bak = root / BACKUP_NAME
        if not bak.exists():
            shutil.copy2(cfg_path, bak)

    engine = existing.setdefault("engine", {})
    locate = engine.setdefault("locate", {})
    for key, value in params.items():
        locate[key] = list(value) if isinstance(value, tuple) else value

    cfg_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cfg_path


def load_existing_locate(root: Path) -> dict:
    """读回当前 ``engine.locate``，用于打 diff。"""
    cfg_path = root / CONFIG_NAME
    if not cfg_path.exists():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return (data.get("engine") or {}).get("locate") or {}


__all__ = ["BACKUP_NAME", "CONFIG_NAME", "apply_locate_params", "load_existing_locate"]
