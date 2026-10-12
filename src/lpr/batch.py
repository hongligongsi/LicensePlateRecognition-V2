"""批量识别的多进程实现。

为什么单独一个模块：进程池要求**提交的函数与返回值都可 pickle**，
因此不能在闭包里抓 ``Recognizer`` / ``Config`` 这类重对象，必须写成
模块级函数 + 可序列化的朴素参数（dict / list / str）。

设计要点
--------
1. **模型只加载一次**：每个子进程在 ``initializer`` 里建好 ``Recognizer``
   并放进进程级缓存；后面的分片复用同一个实例。deep 引擎冷启动约 0.5s，
   若每个文件都重建实例，多进程会比单进程还慢。
2. **分片大于进程数**：分片数取 ``workers * 4``，用少量分片做负载均衡，
   避免某一片恰好都是大图导致整体被拖住。
3. **顺序保证**：分片带回自己的下标，父进程按原始顺序拼回结果，
   与单进程路径的输出顺序完全一致。
4. **逐条隔离**：单个文件失败只在它自己的结果里记 warning，
   不会让整批任务崩掉（与单进程路径行为一致）。
5. **ROI 不回传**：``PlateResult.roi`` 是整块车牌图，跨进程序列化开销极大。
   ``workers > 1`` 时结果里的 ``roi`` 一律为 ``None``；需要 ROI 用 ``workers=1``。
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

from .utils.logging_setup import get_logger

logger = get_logger("batch")

#: 子进程内的识别器缓存（每个进程只有一份）
_WORKER: dict[str, Any] = {}

#: 进程数硬上限：再多的进程只会抢 CPU 与内存，收益为负
MAX_WORKERS = 16

#: 每个子进程至少要分摊到这么多文件才值得并行。
#:
#: 依据：子进程启动 + 重新加载模型是一笔**固定成本**（实测 cv 约 0.5s，
#: deep 约 3~4s：要 import onnxruntime 并重建 OCR 会话）。文件太少时，
#: 这笔固定成本会超过并行省下的时间，并行反而更慢。
#:
#: 阈值按 deep 的固定成本量级取，宁保守勿激进 —— 走串行只是慢一点，
#: 走并行却可能明显更慢，而且更难解释。
MIN_FILES_PER_WORKER = 24


def _cpu_budget() -> int:
    """本进程**实际能用**的 CPU 数。

    ``os.cpu_count()`` 返回的是机器物理核数，在容器 / cgroup / 任务集
    限制下会严重高估（常见于 CI 与容器环境）。Linux 上优先用
    ``sched_getaffinity``，它反映的是真正的调度掩码。
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def resolve_workers(workers: int, n_items: int) -> int:
    """把用户给的 ``workers`` 规范化成真正要开的进程数。

    ``0`` 或不合法值 → 自动取 ``可调度 CPU 数 - 1``（至少 1）；
    最终不会超过待处理文件数与 :data:`MAX_WORKERS`。
    """
    if n_items <= 1:
        return 1
    if workers <= 0:
        workers = max(1, _cpu_budget() - 1)
    return max(1, min(workers, MAX_WORKERS, n_items))


def parallel_is_worthwhile(n_items: int, workers: int) -> bool:
    """文件量是否足以摊平子进程启动 + 模型加载的固定成本。"""
    real = resolve_workers(workers, n_items)
    return real > 1 and n_items >= real * MIN_FILES_PER_WORKER


def _init_worker(engine: str, config_payload: dict) -> None:
    """子进程初始化：重建配置与识别器（每个进程只做一次）。"""
    try:
        from .api import Recognizer
        from .config import Config

        cfg = Config.from_payload(config_payload)
        # 子进程只负责算号码，不需要 ROI 图像，省掉一次拷贝
        cfg.security.keep_roi = False
        _WORKER["rec"] = Recognizer(engine=engine, config=cfg)
    except Exception as exc:  # noqa: BLE001 - 初始化失败要让父进程看到原因
        logger.error("子进程初始化失败: %s", exc)
        _WORKER["error"] = str(exc)


def _run_chunk(job: tuple[int, list[str]]) -> tuple[int, list[dict]]:
    """在子进程里处理一个分片，返回 ``(分片下标, 每个文件的结果字典)``。"""
    index, paths = job
    rec = _WORKER.get("rec")
    out: list[dict] = []
    for path in paths:
        if rec is None:
            out.append(
                {
                    "source": path,
                    "engine": "",
                    "elapsed_ms": 0.0,
                    "warnings": [f"子进程未就绪: {_WORKER.get('error', '未知原因')}"],
                    "plates": [],
                }
            )
            continue
        try:
            # 逐条隔离：单张图异常不能拖垮整批
            out.append(rec.recognize(path).to_dict())
        except Exception as exc:  # noqa: BLE001
            out.append(
                {
                    "source": path,
                    "engine": "",
                    "elapsed_ms": 0.0,
                    "warnings": [f"识别失败: {exc}"],
                    "plates": [],
                }
            )
    return index, out


def make_chunks(sources: Sequence[Any], workers: int) -> list[tuple[int, list[str]]]:
    """把文件列表切成比进程数更多的分片，便于负载均衡。

    单进程（``workers <= 1``）时不做切分，直接返回一个整片 —— 切分对串行
    路径没有任何收益，只会让日志更难读。
    """
    n = len(sources)
    if n == 0:
        return []
    if workers <= 1:
        return [(0, [str(s) for s in sources])]

    n_chunks = max(workers, min(n, workers * 4))
    size = -(-n // n_chunks)  # 向上取整
    chunks: list[tuple[int, list[str]]] = []
    for start in range(0, n, size):
        piece = [str(s) for s in sources[start : start + size]]
        chunks.append((len(chunks), piece))
    return chunks


def run_parallel(
    sources: Sequence[Any],
    engine: str,
    config_payload: dict,
    workers: int,
) -> list[dict]:
    """并行识别，返回**与输入同序**的结果字典列表。

    一个文件一个结果字典，字段与 ``RecognizeOutcome.to_dict()`` 对齐。
    父进程负责按隐私策略脱敏，子进程一律输出明文（只脱敏一次）。
    """
    if not sources:
        return []

    workers = resolve_workers(workers, len(sources))
    chunks = make_chunks(sources, workers)
    logger.info(
        "并行批量识别: %d 个文件 / %d 进程 / %d 分片", len(sources), workers, len(chunks)
    )

    # spawn：Windows 上默认即 spawn；显式指定可避免 Linux 上 fork 出
    # 持有全局锁（INFERENCE_LOCK）的副本导致死锁
    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    collected: dict[int, list[dict]] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(engine, config_payload),
    ) as pool:
        for index, part in pool.map(_run_chunk, chunks):
            collected[index] = part

    ordered: list[dict] = []
    for index in sorted(collected):
        ordered.extend(collected[index])
    return ordered
