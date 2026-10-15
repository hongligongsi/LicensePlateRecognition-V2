"""三合一引擎：把 ``cv`` / ``deep`` / ``yolo`` 编排成一个引擎。

三种策略（``EngineConfig.fusion_strategy`` 切换）：

* ``ensemble``（默认）**并联投票融合** —— 所有可用成员各跑一次，把落在
  同一处的框分组，组内按 ``引擎权重 × 置信度`` 给号码投票，票高者胜出。
  三个引擎的错误模式互补（cv 擅长清晰正视角、deep 擅长中文与模糊、
  yolo 擅长侧倾透视），所以融合后的准确率与召回都高于任一单引擎。
  代价是耗时≈三者相加。
* ``chain`` **顺序兜底链** —— 按成员顺序依次尝试，一旦拿到"够可信"的结果
  就停（``fusion_chain_accept``）。耗时≈最快命中的那个引擎，只提召回不提
  单次准确率。
* ``auto`` **特征选优** —— 先按图像特征（长边分辨率、模糊度）给成员排序，
  再按序尝试、命中即止。清晰大图走 deep（快且准），小图/糊图走 yolo
  （检测网络更稳），不可用就顺延。

三个共同点：

* **成员按可用性退席**：某个成员缺依赖或缺权重，只是少一票，不影响其余
  成员工作；全部成员都不可用才抛 :class:`EngineUnavailable`。
* **不给下游换语义**：融合结果的 ``confidence`` 仍是**胜出成员的原始置信度**
  （不是票占比），这样调用方的阈值、跟踪器投票权重等逻辑不需要为融合引擎
  单独适配；融合特有信息放在 ``meta["fusion"]`` 里。
* **保留无读数的框**：检测到了但没读出来的候选仍然保留（``text=""``），
  与 ``yolo`` 引擎的约定一致 —— 静默丢框会让人误以为模型坏了。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import EngineConfig
from ..plate_rules import plate_score
from ..types import PlateResult
from ..utils.image import imread
from ..utils.logging_setup import get_logger
from .base import BaseEngine, EngineUnavailable

#: 策略取值
STRATEGY_ENSEMBLE = "ensemble"
STRATEGY_CHAIN = "chain"
STRATEGY_AUTO = "auto"
STRATEGIES: tuple[str, ...] = (STRATEGY_ENSEMBLE, STRATEGY_CHAIN, STRATEGY_AUTO)

#: 默认成员（顺序即 chain 的尝试顺序、也是 auto 同档位时的先后）
DEFAULT_MEMBERS: tuple[str, ...] = ("yolo", "deep", "cv")

#: 默认票权：检测网络与深度学习引擎各 1.0，传统 CV 0.6
#: （cv 在清晰正视角下很可靠，但整体错误率高于另两个，给低一点的票权）
DEFAULT_ENGINE_WEIGHTS: dict[str, float] = {"yolo": 1.0, "deep": 1.0, "cv": 0.6}

#: 同一个车牌的分组阈值：两框 IoU 超过它就算落在同一处
DEFAULT_GROUP_IOU = 0.4

#: 一组结果要**留下**的门槛：至少 ``fusion_min_support`` 个成员投它，
#: 或票权达到 ``fusion_trusted_weight``（单成员独存）。
#:
#: 这条规则不是拍脑袋来的，是实测逼出来的：``cv`` 引擎在 4096×2304 的真实
#: 样张上会给出 9 条候选（其中 7 条是误检），而 ``yolo`` 给 2 条全中、
#: ``deep`` 给 1 条全中。若融合只做并集，精度会被最弱的成员拖垮 ——
#: 于是"cv 独有"的组必须凑到第二个成员的票才能存活（cv 票权 0.6 < 1.0）。
#: 想取并集求最大召回，把 ``fusion_min_support`` 设成 1 即可。
DEFAULT_MIN_SUPPORT = 2
DEFAULT_TRUSTED_WEIGHT = 1.0

#: ``chain`` 的"够可信"门槛
DEFAULT_CHAIN_ACCEPT = 0.75

#: 票权下限：有的引擎会给出 text 但 confidence 为 0，若直接按 0 计权，
#: 它的票永远赢不了，等于白跑一次。给一个很小的底，保证"只要有读数就有票"。
VOTE_MIN_CONFIDENCE = 0.05

#: ``auto`` 的判据阈值
AUTO_SMALL_SIDE = 480
AUTO_BLUR_THRESHOLD = 60.0
#: 算模糊度时先把图缩到这一长边，避免 4K 图上做 Laplacian 白等
AUTO_BLUR_PROBE_SIDE = 640

#: 融合引擎自己的名字（成员里不允许出现它，否则递归）
SELF_NAME = "fusion"
#: 策略别名，不是具体引擎，不能当成员
NON_MEMBER_NAMES = frozenset({SELF_NAME, "cascade", "auto"})

logger = get_logger("engines.fusion")


# ---------------------------------------------------------------------- 图像特征


def blur_score(image: np.ndarray) -> float:
    """拉普拉斯方差：越大越清晰。

    这是判断"要不要动用检测网络"的常用指标（越小越糊）。为了不让 4K 图
    白等，先缩到 :data:`AUTO_BLUR_PROBE_SIDE` 再算。
    """
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side > AUTO_BLUR_PROBE_SIDE:
        scale = AUTO_BLUR_PROBE_SIDE / long_side
        gray = cv2.resize(
            gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_single_engine(
    image: np.ndarray,
    members: tuple[str, ...] | list[str],
) -> tuple[str, str]:
    """``auto`` 策略的排序器：按图像特征把成员排成一个"优先尝试顺序"。

    规则是有意写得简单可解释的（阈值都是显式常量，不玩玄学）：

    * **小图或糊图** → 检测网络优先。车牌只占几个像素、或边缘已经糊掉时，
      轮廓法与 OCR 都容易失效，而检测网络是靠全局语义找车牌的。
    * **较大且清晰的图** → ``deep`` 优先（定位引导 OCR，速度与精度平衡最好）。

    返回 ``(首选, 理由)``；不可用的成员不在 ``members`` 里，所以这里不做
    可用性判断。
    """
    ordered = list(members)
    if not ordered:
        raise ValueError("没有可用成员")

    side = max(image.shape[:2]) if image is not None and image.size else 0
    blur = blur_score(image)
    prefer_detector = side < AUTO_SMALL_SIDE or blur < AUTO_BLUR_THRESHOLD
    preference = ("yolo", "deep", "cv") if prefer_detector else ("deep", "yolo", "cv")
    ranked = [m for m in preference if m in ordered]
    ranked += [m for m in ordered if m not in ranked]

    why = (
        f"图像较小或偏糊（长边={side}，模糊度={blur:.1f}）→ 检测网络优先"
        if prefer_detector
        else f"图像清晰且够大（长边={side}，模糊度={blur:.1f}）→ 深度学习引擎优先"
    )
    return ranked[0], why


def _rank_order(members: tuple[str, ...]) -> list[str]:
    """全局优先级，用于平票时的裁决（与 chain 的默认顺序一致）。"""
    ranked = [m for m in DEFAULT_MEMBERS if m in members]
    return ranked + [m for m in members if m not in ranked]


# ---------------------------------------------------------------------- 分组与投票


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """``(x, y, w, h)`` 框的 IoU。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _bbox4(bbox: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    """把成员的框原样搬成 4 元组（顺手把 numpy 标量收成 int）。"""
    if not bbox:
        return None
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


def group_results(results: list[PlateResult], iou_threshold: float = DEFAULT_GROUP_IOU):
    """把多个引擎的候选按"是否同一块车牌"分组。

    判定：框有重叠且 IoU 超过阈值，或号码完全相同。没有框的候选只能靠
    号码相同归组，否则各自成组（宁可不合并，也不要把两块车牌揉成一团 ——
    揉错了会把两个号码合成一个，比漏合并严重得多）。

    先按"有读数的、置信度高的"排前面，让它们决定组心，避免低质量候选
    把组心带偏。
    """
    ordered = sorted(
        results,
        key=lambda r: (not r.text, -r.confidence, -plate_score(r.text)),
    )
    groups: list[list[PlateResult]] = []
    for res in ordered:
        placed = False
        for group in groups:
            head = group[0]
            if res.text and res.text == head.text:
                group.append(res)
                placed = True
                break
            if res.bbox and head.bbox and _iou(res.bbox, head.bbox) >= iou_threshold:
                group.append(res)
                placed = True
                break
        if not placed:
            groups.append([res])
    return groups


def vote_group(
    group: list[PlateResult],
    engine_weights: dict[str, float],
    order: list[str],
) -> tuple[str, dict[str, float]]:
    """组内投票，返回 ``(胜出号码, {号码: 票数})``。

    每个引擎在组内**只投一票**（同一引擎对同一块车牌往往给出多个候选，
    都算票会让话多的引擎主导结果），取它置信度最高的那个读数。

    票权 = ``引擎权重 × max(置信度, VOTE_MIN_CONFIDENCE)``。
    """
    best_per_engine: dict[str, PlateResult] = {}
    for res in group:
        if not res.text:
            continue
        key = res.engine or "unknown"
        prev = best_per_engine.get(key)
        if prev is None or res.confidence > prev.confidence:
            best_per_engine[key] = res

    votes: dict[str, float] = {}
    for engine, res in best_per_engine.items():
        weight = engine_weights.get(engine, 1.0) * max(res.confidence, VOTE_MIN_CONFIDENCE)
        votes[res.text] = votes.get(res.text, 0.0) + weight

    if not votes:
        return "", {}

    rank = {name: i for i, name in enumerate(order)}
    winner = max(
        votes,
        key=lambda text: (
            votes[text],
            plate_score(text),
            len(text),
            -min(
                rank.get(best_per_engine[e].engine, 99)
                for e in best_per_engine
                if best_per_engine[e].text == text
            ),
        ),
    )
    return winner, votes


def _representative(group: list[PlateResult], text: str) -> PlateResult:
    """挑出代表该结果的实例：优先有该号码、置信度最高者。"""
    candidates = [r for r in group if r.text == text]
    if not candidates:
        candidates = group
    return max(candidates, key=lambda r: (r.confidence, plate_score(r.text)))


def fuse_group(
    group: list[PlateResult],
    engine_weights: dict[str, float],
    order: list[str],
    *,
    strategy: str,
) -> PlateResult:
    """把一个分组合成一个 :class:`PlateResult`。

    ``confidence`` 用**胜出成员的原始置信度**（见模块文档）；票数、一致度、
    各成员读数都进 ``meta["fusion"]``，方便定位"是谁把号码带偏的"。
    """
    winner_text, votes = vote_group(group, engine_weights, order)
    rep = _representative(group, winner_text)
    total = sum(votes.values())
    winner_votes = votes.get(winner_text, 0.0)

    contributors: dict[str, dict[str, Any]] = {}
    for res in group:
        key = res.engine or "unknown"
        prev = contributors.get(key)
        if prev is None or res.confidence > prev["confidence"]:
            contributors[key] = {
                "text": res.text,
                "confidence": round(float(res.confidence), 4),
                "bbox": list(res.bbox) if res.bbox else None,
            }

    merged = PlateResult(
        text=winner_text,
        color=rep.color,
        layout=rep.layout,
        bbox=_bbox4(rep.bbox),
        roi=rep.roi,
        confidence=float(rep.confidence),
        engine=SELF_NAME,
        char_confidences=list(rep.char_confidences),
        meta=dict(rep.meta),
    )
    merged.meta["fusion"] = {
        "strategy": strategy,
        "members": sorted({r.engine or "unknown" for r in group}),
        "votes": {text: round(w, 4) for text, w in sorted(votes.items(), key=lambda kv: -kv[1])},
        "agreement": round(winner_votes / total, 4) if total else 0.0,
        "support": sum(1 for c in contributors.values() if c["text"] == winner_text),
        "contributors": contributors,
    }
    if not winner_text:
        # 与 yolo 引擎同一约定：检到了但没读出来，保框留白而不是静默丢框
        merged.meta["ocr_failed"] = True
    return merged


def fuse_results(
    results: list[PlateResult],
    engine_weights: dict[str, float] | None = None,
    *,
    members: tuple[str, ...] = DEFAULT_MEMBERS,
    iou_threshold: float = DEFAULT_GROUP_IOU,
    strategy: str = STRATEGY_ENSEMBLE,
    min_support: int = DEFAULT_MIN_SUPPORT,
    trusted_weight: float = DEFAULT_TRUSTED_WEIGHT,
    dropped: list[dict[str, Any]] | None = None,
) -> list[PlateResult]:
    """并联融合的纯函数入口（不依赖引擎实例，便于单测）。

    Args:
        dropped: 传入一个 list 时，被准入规则滤掉的组会写进去（诊断用）。
            丢弃**不能是静默的** —— 否则用户会以为引擎漏检，其实是策略滤掉了。
    """
    if not results:
        return []
    weights = dict(DEFAULT_ENGINE_WEIGHTS if engine_weights is None else engine_weights)
    order = _rank_order(members)
    # 只有一个成员在岗时，"至少两票"永远凑不齐 —— 自动放宽，否则会全滤成空
    effective_min_support = max(1, min(min_support, len(members)))

    merged: list[PlateResult] = []
    for group in group_results(results, iou_threshold):
        candidate = fuse_group(group, weights, order, strategy=strategy)
        info = candidate.meta.get("fusion", {})
        if not _passes_admission(info, weights, effective_min_support, trusted_weight):
            if dropped is not None:
                dropped.append(
                    {
                        "text": candidate.text,
                        "votes": info.get("votes", {}),
                        "support": info.get("support", 0),
                        "bbox": list(candidate.bbox) if candidate.bbox else None,
                    }
                )
            continue
        merged.append(candidate)

    merged.sort(key=lambda r: (-plate_score(r.text), -r.confidence, r.bbox or (0, 0, 0, 0)))
    return merged


def _passes_admission(
    info: dict[str, Any],
    weights: dict[str, float],
    min_support: int,
    trusted_weight: float,
) -> bool:
    """准入判定：够票数，或来自足够可信的成员。

    没有读数（``text == ""``）的组**一律保留** —— 那是"检到了但没认出来"，
    属于要给人看的信息，不是误检噪声。
    """
    votes: dict[str, float] = info.get("votes", {})
    if not votes:
        return True
    winner_weight = max(votes.values())
    if winner_weight >= trusted_weight:
        return True
    return int(info.get("support", 0)) >= min_support


# ---------------------------------------------------------------------- 引擎


class FusionEngine(BaseEngine):
    """三合一引擎（``cv`` + ``deep`` + ``yolo``）。

    成员引擎是**共享单例**（走 ``get_engine``），因此三个引擎的模型只加载
    一次，切回单引擎模式不会重复付出加载开销。本引擎 ``close()`` 只丢自己
    的引用，**不会**关掉成员 —— 否则会顺手把别的引擎砸了。
    """

    name = SELF_NAME
    display_name = "三合一引擎"
    description = (
        "同时编排 cv / deep / yolo：并联投票融合（默认）/ 顺序兜底链 / 特征选优，"
        "缺权重或依赖的成员自动退席；融合结果保留各成员读数与票数明细"
    )

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        strategy: str | None = None,
        members: tuple[str, ...] | list[str] | None = None,
        engine_weights: dict[str, float] | None = None,
        group_iou: float | None = None,
        chain_accept: float | None = None,
        min_support: int | None = None,
        trusted_weight: float | None = None,
        member_factory: Callable[[str], Any] | None = None,
    ) -> None:
        """
        Args:
            config: ``EngineConfig``；``fusion_*`` 字段是本引擎的配置来源。
            strategy: ``ensemble`` / ``chain`` / ``auto``。显式传入时**覆盖**
                config（优先级：实参 > config > 引擎默认）。
            members: 成员引擎名及顺序。
            engine_weights: 各成员票权。
            group_iou: 同一车牌的分组 IoU 阈值。
            chain_accept: ``chain`` 的"够可信"门槛。
            min_support: 融合结果要留下所需的最少一致成员数。
            trusted_weight: 单成员独存所需的票权。
            member_factory: 成员引擎工厂（默认 ``get_engine``）。测试注入用。
        """
        cfg = config
        super().__init__()
        self.config = config
        self.strategy = strategy or _cfg_value(cfg, "fusion_strategy", STRATEGY_ENSEMBLE)
        raw_members = members if members is not None else _cfg_value(cfg, "fusion_members", None)
        self.members = validate_members(raw_members or DEFAULT_MEMBERS)
        raw_weights = (
            engine_weights
            if engine_weights is not None
            else _cfg_value(cfg, "fusion_engine_weights", None)
        )
        self.engine_weights = dict(raw_weights or DEFAULT_ENGINE_WEIGHTS)
        self.group_iou = float(
            group_iou
            if group_iou is not None
            else _cfg_value(cfg, "fusion_group_iou", DEFAULT_GROUP_IOU)
        )
        self.chain_accept = float(
            chain_accept
            if chain_accept is not None
            else _cfg_value(cfg, "fusion_chain_accept", DEFAULT_CHAIN_ACCEPT)
        )
        self.min_support = int(
            min_support
            if min_support is not None
            else _cfg_value(cfg, "fusion_min_support", DEFAULT_MIN_SUPPORT)
        )
        self.trusted_weight = float(
            trusted_weight
            if trusted_weight is not None
            else _cfg_value(cfg, "fusion_trusted_weight", DEFAULT_TRUSTED_WEIGHT)
        )
        if self.strategy not in STRATEGIES:
            raise ValueError(f"未知融合策略: {self.strategy}（可选: {', '.join(STRATEGIES)}）")
        self._member_factory = member_factory
        #: 已就绪的成员：名 → 引擎实例
        self._members: dict[str, Any] = {}
        #: 成员退席原因：名 → 原因（诊断用）
        self.member_errors: dict[str, str] = {}
        #: 最近一次 ensemble 被准入规则滤掉的孤立候选（诊断用，见 fuse_results）
        self.last_dropped: list[dict[str, Any]] = []
        self._ready = False

    # ------------------------------------------------------------- 生命周期

    def load(self) -> None:
        """加载所有**能加载的**成员；一个都上不来才报不可用。

        这里不去看 ``is_available()`` 的类方法探测结果，而是直接试着建实例：
        探测只看默认权重目录，用户把权重放到自定义目录时探测会说"不可用"，
        真去加载反而是成的 —— 试错比预判准。
        """
        self._members = {}
        self.member_errors = {}
        for name in self.members:
            try:
                engine = self._resolve_member(name)
                engine.ensure_ready()
            except Exception as exc:  # 成员坏了不该拖垮整个融合引擎
                reason = f"{type(exc).__name__}: {exc}"
                self.member_errors[name] = reason
                logger.warning("融合成员 %s 不可用，本次退席（%s）", name, reason)
                continue
            self._members[name] = engine
            logger.info("融合成员 %s 就绪", name)

        if not self._members:
            detail = "；".join(f"{n}: {r}" for n, r in self.member_errors.items()) or "未配置成员"
            raise EngineUnavailable(f"三合一引擎没有可用成员（{detail}）")
        self._ready = True
        logger.info(
            "三合一引擎就绪 | 策略=%s | 成员=%s%s",
            self.strategy,
            ",".join(self._members),
            f" | 退席={','.join(self.member_errors)}" if self.member_errors else "",
        )

    def close(self) -> None:
        """只丢自己的成员引用 —— 成员是共享单例，不能在这里关掉。"""
        self._members = {}
        self._ready = False

    def _resolve_member(self, name: str) -> Any:
        if self._member_factory is not None:
            return self._member_factory(name)
        from .registry import get_engine

        return get_engine(name, self.config)

    # ------------------------------------------------------------- 可用性

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """至少要有一个成员可用（只看默认配置下的默认权重目录）。"""
        from .registry import _REGISTRY  # 延迟导入：避免导入期循环依赖

        ok_members: list[str] = []
        details: list[str] = []
        for name in DEFAULT_MEMBERS:
            engine_cls = _REGISTRY.get(name)
            if engine_cls is None:
                details.append(f"{name}: 未注册")
                continue
            ok, reason = engine_cls.is_available()
            if ok:
                ok_members.append(name)
            else:
                details.append(f"{name}: {reason}")
        if ok_members:
            return True, f"可用成员: {', '.join(ok_members)}"
        return False, "无可用成员（" + "；".join(details) + "）"

    # ------------------------------------------------------------- 识别

    def predict(self, image: np.ndarray | str | Path) -> list[PlateResult]:
        """按当前策略识别一张图。"""
        self.ensure_ready()
        img = image if isinstance(image, np.ndarray) else imread(image)
        if self.strategy == STRATEGY_CHAIN:
            return self._predict_chain(img)
        if self.strategy == STRATEGY_AUTO:
            return self._predict_auto(img)
        return self._predict_ensemble(img)

    def predict_path(self, path: str | Path) -> list[PlateResult]:
        return self.predict(path)

    # ------------------------------------------------------------- 三种策略

    def _predict_ensemble(self, img: np.ndarray) -> list[PlateResult]:
        """并联：全都跑，然后按空间分组投票，再按准入规则过滤。"""
        collected: list[PlateResult] = []
        elapsed: dict[str, float] = {}
        for name, engine in self._members.items():
            started = time.perf_counter()
            try:
                collected.extend(engine.predict(img) or [])
            except Exception as exc:  # 单个成员崩了不影响其它成员
                self.member_errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("融合成员 %s 本次调用失败，跳过（%s）", name, exc)
                continue
            elapsed[name] = round((time.perf_counter() - started) * 1000, 1)
        if not collected:
            self.last_dropped = []
            return []

        dropped: list[dict[str, Any]] = []
        fused = fuse_results(
            collected,
            self.engine_weights,
            members=tuple(self._members),
            iou_threshold=self.group_iou,
            strategy=self.strategy,
            min_support=self.min_support,
            trusted_weight=self.trusted_weight,
            dropped=dropped,
        )
        self.last_dropped = dropped
        if dropped:
            # 丢弃必须可见：否则用户会以为引擎漏检，其实是准入规则滤掉了
            logger.info(
                "融合准入滤掉 %d 个孤立候选（未达 %d 票且票权 < %.2f）：%s",
                len(dropped),
                self.min_support,
                self.trusted_weight,
                ", ".join(str(d.get("text") or "(无读数)") for d in dropped),
            )
        for res in fused:
            info = res.meta.setdefault("fusion", {})
            info["elapsed_ms"] = elapsed
            if self.member_errors:
                info["skipped"] = dict(self.member_errors)
            if dropped:
                info["dropped_count"] = len(dropped)
        return fused

    def _predict_chain(self, img: np.ndarray) -> list[PlateResult]:
        """顺序兜底：够可信就停，否则继续下一个；全都不够就返回最好的那档。"""
        best: list[PlateResult] | None = None
        best_key = (-1.0, -1.0)
        tried: list[str] = []
        for name, engine in self._members.items():
            tried.append(name)
            try:
                results = engine.predict(img) or []
            except Exception as exc:
                self.member_errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("融合成员 %s 本次调用失败，跳过（%s）", name, exc)
                continue
            if results and _is_confident(results, self.chain_accept):
                return _stamp(
                    results, strategy=self.strategy, tried=tried, decided=name, self_name=self.name
                )
            key = _chain_key(results)
            if key > best_key:
                best, best_key = results, key
        if best is None:
            return []
        return _stamp(
            best, strategy=self.strategy, tried=tried, decided="best-effort", self_name=self.name
        )

    def _predict_auto(self, img: np.ndarray) -> list[PlateResult]:
        """特征选优：先按图像特征排序，再按序尝试、命中即止。"""
        names = tuple(self._members)
        if not names:
            return []
        first, why = select_single_engine(img, names)
        preference = [first, *[m for m in _rank_order(names) if m != first]]
        tried: list[str] = []
        for name in preference:
            engine = self._members.get(name)
            if engine is None:
                continue
            tried.append(name)
            try:
                results = engine.predict(img) or []
            except Exception as exc:
                self.member_errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("融合成员 %s 本次调用失败，跳过（%s）", name, exc)
                continue
            if results:
                return _stamp(
                    results,
                    strategy=self.strategy,
                    tried=tried,
                    decided=name,
                    self_name=self.name,
                    reason=why,
                )
        return []


def _stamp(
    results: list[PlateResult],
    *,
    strategy: str,
    tried: list[str],
    decided: str,
    self_name: str,
    reason: str = "",
) -> list[PlateResult]:
    """给"没有融合、直接透传某成员结果"的路由结果打上来源标记。"""
    for res in results:
        res.engine = self_name
        info = res.meta.setdefault("fusion", {})
        info.update({"strategy": strategy, "tried": list(tried), "decided": decided})
        if reason:
            info["reason"] = reason
    return results


def _is_confident(results: list[PlateResult], threshold: float) -> bool:
    """是否存在"够可信"的读数。"""
    return any(r.text and r.confidence >= threshold for r in results)


def _chain_key(results: list[PlateResult]) -> tuple[float, float]:
    """比较两档结果的优劣：先比有几个读数，再比平均置信度。"""
    texts = [r.text for r in results if r.text]
    if not texts:
        return (-1.0, -1.0)
    mean_conf = sum(r.confidence for r in results if r.text) / len(texts)
    return (float(len(texts)), float(mean_conf))


def validate_members(members: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """校验成员名单：非空、不重复、都是具体引擎、且不含自己。"""
    from .registry import available_engines  # 延迟导入：避免导入期循环依赖

    known = set(available_engines())
    out: list[str] = []
    for name in members:
        if name in NON_MEMBER_NAMES:
            raise ValueError(f"{name!r} 不能作为融合成员（策略名/自己的名字不能当成员）")
        if name not in known:
            raise ValueError(f"未知融合成员: {name!r}（已注册引擎: {', '.join(sorted(known))}）")
        if name in out:
            raise ValueError(f"融合成员重复: {name!r}")
        out.append(name)
    if not out:
        raise ValueError("融合成员不能为空")
    return tuple(out)


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    """安全地取 config 字段（config 可能是 None 或测试替身）。"""
    if cfg is None:
        return default
    return getattr(cfg, name, default)
