"""真实样张行为回归基线。

这是"改了定位参数 / 动了定位算法，有没有悄悄改变既有样张行为"的自动拦截。

为什么用快照而不是准确率
------------------------
带标注的真实数据集不适合进仓库（体积 + 版权），而**合成图不能当真值**
（实测合成牌只有占到画面一半宽才勉强被定位到，且找到的是字符区而非车牌区，
拿它算准确率是自欺欺人）。所以这里的基线是真实样张上的**行为快照**：

* 引擎用确定性的 ``cv``（SVM，纯 CPU，不下载任何模型）→ 结果可复现
* 记录每张图的候选数、top 候选框、top 候选底色
* 判据带容差（IoU ≥ 0.85、候选数 ±1），因为不同 OpenCV 版本的形态学
  实现可能有 1~2 像素级差异，那不算回归

覆盖范围
--------
快照覆盖 ``img/`` + ``samples/`` 下**全部**图片。v2.7.3 时为 403 张
（``img/`` 401 + ``samples/`` 2），可读性无一异常。

注意 ``img/`` 同时被当作下载样本的落图目录（``tools/fetch_ccpd.py`` 下到
``datasets/ccpd_raw``，之后常被搬进 ``img/``），于是**加图就会让
:func:`test_baseline_covers_repo_samples` 变红** —— 那是故意的绊线：新增样张
必须重新生成基线，否则下面几条逐张对照会失去意义。体积护栏见
:data:`SAMPLE_SIZE_BUDGET_MB`。

基线更新方式：``python tools/make_regression_baseline.py --write``
（**确认行为变化是预期之内**才更新）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lpr.config import Config  # noqa: E402
from lpr.engines.locate import locate_plates  # noqa: E402

BASELINE_PATH = ROOT / "tests" / "baselines" / "real_samples.json"

#: top 候选框允许的最小 IoU（低于它才判为行为漂移）
MIN_IOU = 0.85
#: 候选数允许的浮动（±1 个，容忍形态学实现的微小差异）
CAND_TOLERANCE = 1


def _xywh_to_xyxy(box: list[int]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(f"缺少基线文件 {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict:
    """当前行为快照（复用工具脚本，保证两边口径完全一致）。"""
    sys.path.insert(0, str(ROOT / "tools"))
    from make_regression_baseline import snapshot

    return snapshot()


def _by_path(records: list[dict]) -> dict[str, dict]:
    """按路径建索引。逐张对照必须靠它配对，不能靠位置。"""
    return {rec["path"]: rec for rec in records}


def test_baseline_covers_repo_samples(baseline, current):
    """基线与当前快照必须覆盖同一批样张，否则等于没测。

    这条**故意**对新增样张也报错：加了图就要重新生成基线，不然下面几条逐张
    对照会失去意义。所以错误信息里直接把多出来/缺了哪些列出来，省得自己 diff。
    """
    want = [r["path"] for r in baseline["records"]]
    got = [r["path"] for r in current["records"]]
    assert want, "基线里一条记录都没有"
    if want != got:
        added = sorted(set(got) - set(want))
        removed = sorted(set(want) - set(got))
        detail = []
        if added:
            detail.append(f"多出 {len(added)} 张（前 5 张）：{added[:5]}")
        if removed:
            detail.append(f"缺少 {len(removed)} 张（前 5 张）：{removed[:5]}")
        pytest.fail(
            "样张清单变了（新增/删除了样张？确认无误后重新生成基线："
            "`python tools/make_regression_baseline.py --write`）：\n  " + "\n  ".join(detail)
        )


def test_all_repo_samples_readable(current):
    """仓库自带样张必须全部可读（防止有人提交了坏文件）。"""
    broken = [r["path"] for r in current["records"] if not r.get("readable")]
    assert not broken, f"这些样张读不出来: {broken}"


def test_candidate_presence_matches_baseline(baseline, current):
    """每张样张"定位得到候选 / 定位不到"必须与基线一致。

    判据是**逐张比较有/无**，而不是"断言每张都能定位到"：403 张里本来就有
    2 张 cv 引擎给不出候选（``img/0315_067.jpg``、``img/sample_1095.jpg``），
    断言"全都能定位"会把这 2 张误判成回归。而定位器整体失效的表现是
    **大批原本有候选的样张同时落空**，逐张比较一样拦得住。

    这条同时补上 :meth:`TestBehaviorUnchanged.test_top_bbox_not_drifted` 的缺口：
    那条在"基线有框、现在没框"时会 ``continue`` 跳过，只靠它抓不住
    "某张图从有候选变成没候选"。
    """
    now_by_path = _by_path(current["records"])
    flipped = []
    for base in baseline["records"]:
        now = now_by_path.get(base["path"])
        if now is None or not now.get("readable"):
            continue
        was, is_now = bool(base.get("top_bbox")), bool(now.get("top_bbox"))
        if was != is_now:
            flipped.append(f"{base['path']}: 有候选 {was} -> {is_now}")
    assert not flipped, (
        "这些样张「有没有候选」与基线不一致（若为预期改动，请重新生成基线）：\n  "
        + "\n  ".join(flipped)
    )


class TestBehaviorUnchanged:
    """逐张对照基线与当前行为。

    必须按**路径**配对，不能按位置 ``zip`` —— 位置配对只要样张数量一变，
    就变成"图 A 的基线" 比 "图 B 的现状"，报出一屏假漂移把真信号埋掉。
    （v2.7.3 开发期实测踩到：``img/`` 被塞进 193 张本地图后，基线里那 11 张
    真样张全部报 IoU≈0，而按路径逐张复算的 IoU 全是 1.000。）
    """

    def test_top_bbox_not_drifted(self, baseline, current):
        now_by_path = _by_path(current["records"])
        drifted = []
        for base in baseline["records"]:
            now = now_by_path.get(base["path"])
            if now is None or not base.get("top_bbox") or not now.get("top_bbox"):
                continue
            iou = _iou(base["top_bbox"], now["top_bbox"])
            if iou < MIN_IOU:
                drifted.append(
                    f"{base['path']}: IoU={iou:.3f} 基线={base['top_bbox']} 现在={now['top_bbox']}"
                )
        assert not drifted, (
            "top 候选框发生漂移（若为预期改动，请重新生成基线）：\n  " + "\n  ".join(drifted)
        )

    def test_candidate_count_not_drifted(self, baseline, current):
        now_by_path = _by_path(current["records"])
        drifted = []
        for base in baseline["records"]:
            now = now_by_path.get(base["path"])
            if now is None:
                continue
            delta = abs(base["n_candidates"] - now["n_candidates"])
            if delta > CAND_TOLERANCE:
                drifted.append(f"{base['path']}: {base['n_candidates']} -> {now['n_candidates']}")
        assert not drifted, "候选数量发生明显变化：\n  " + "\n  ".join(drifted)

    def test_top_color_not_drifted(self, baseline, current):
        now_by_path = _by_path(current["records"])
        drifted = []
        for base in baseline["records"]:
            now = now_by_path.get(base["path"])
            if now is None or base.get("top_color") == now.get("top_color"):
                continue
            drifted.append(f"{base['path']}: {base.get('top_color')} -> {now.get('top_color')}")
        assert not drifted, "top 候选底色判定变了：\n  " + "\n  ".join(drifted)


def test_baseline_is_documented_as_snapshot(baseline):
    """基线文件必须自述"这是快照、不是准确率"，避免被误读。"""
    note = baseline.get("_note", "")
    assert "快照" in note
    assert "不是准确率" in note or "非准确率" in note


def test_baseline_note_names_the_update_tool(baseline):
    assert "make_regression_baseline" in baseline.get("_note", "")


def test_locate_is_deterministic_on_real_sample():
    """同一张图跑两遍必须完全一致（快照基线成立的前提）。"""
    from make_regression_baseline import _imread

    path = ROOT / "img" / "double_yellow.jpg"
    if not path.exists():
        pytest.skip("样张缺失")
    img = _imread(path)
    if img is None:
        pytest.skip("样张不可读")

    cfg = Config.load().engine.locate
    first, _ = locate_plates(img, cfg, None)
    second, _ = locate_plates(img, cfg, None)

    assert [c.bbox for c in first] == [c.bbox for c in second]
    assert [c.color for c in first] == [c.color for c in second]


def test_tolerance_is_not_meaningless():
    """容差不能大到"怎么变都通过"：明显不同的框必须被判为漂移。"""
    assert _iou([0, 0, 100, 40], [0, 0, 100, 40]) == pytest.approx(1.0)
    assert _iou([0, 0, 100, 40], [200, 200, 100, 40]) == 0.0
    assert _iou([0, 0, 100, 40], [10, 10, 100, 40]) < MIN_IOU


def test_imread_helper_handles_bad_file(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    from make_regression_baseline import _imread

    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    assert _imread(broken) is None


def test_snapshot_skips_unreadable_without_crashing(tmp_path, monkeypatch):
    """坏文件不能让快照生成崩掉，而要记 readable=False。"""
    sys.path.insert(0, str(ROOT / "tools"))
    import make_regression_baseline as mrb

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"xxxx")
    monkeypatch.setattr(mrb, "collect", lambda: [bad])

    snap = mrb.snapshot()
    assert snap["records"][0]["readable"] is False


#: 样张总体积护栏（``img/`` + ``samples/``）。
#:
#: **40MB → 150MB（2026-10-16 调整）**：40MB 那个数字是 ``img/`` 里只有 9 张
#: 入库样张时定的，当时它等价于"CI checkout 会很快"。现在 ``img/`` 同时是
#: CCPD 下载样本的落图目录（403 张 / 75.7MB），用户决定保留入库、按现实改门槛，
#: 于是这个断言退化成**失控增长护栏**：它拦的是"哪天不小心把几百 MB 塞进来"，
#: 不再是"体积很小 / checkout 很快"的承诺。想让体积回到原来的量级，应当把
#: 下载图放进 ``datasets/``（已在 .gitignore 里），而不是继续往 ``img/`` 里搬。
SAMPLE_SIZE_BUDGET_MB = 150


def test_repo_samples_are_small_enough_for_ci():
    """样张总体积要有护栏，防止某天把几百 MB 顺手塞进仓库。

    这个上限现在只防失控，不代表体积小 —— 取舍的来龙去脉见
    :data:`SAMPLE_SIZE_BUDGET_MB` 的注释。
    """
    total = 0
    for d in ("img", "samples"):
        root = ROOT / d
        if root.is_dir():
            total += sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    budget = SAMPLE_SIZE_BUDGET_MB * 1024 * 1024
    assert total < budget, (
        f"样张总体积 {total / 1048576:.1f}MB，超过 {SAMPLE_SIZE_BUDGET_MB}MB 护栏"
    )


def test_cv_engine_baseline_needs_no_download():
    """基线必须能在没有网络的 CI 上跑 —— 即不依赖 rapidocr/paddle。"""
    src = (ROOT / "tools" / "make_regression_baseline.py").read_text(encoding="utf-8")
    assert "rapidocr" not in src
    assert "paddleocr" not in src
    assert "recognize_many" not in src  # 不走深度引擎


def test_iou_helper_is_xywh_aware():
    """IoU 必须按 (x,y,w,h) 口径算 —— 本项目全项目统一该口径。"""
    assert _iou([10, 20, 100, 50], [10, 20, 100, 50]) == pytest.approx(1.0)
    # 若是按 xyxy 误读，[10,20,100,50] 会变成 (10,20)-(100,50)
    assert _iou([10, 20, 100, 50], [110, 20, 100, 50]) == 0.0


def test_numpy_and_cv2_available():
    """基线测试的前置依赖（opencv 是基础依赖，不该缺）。"""
    assert cv2.__version__
    assert np.zeros((2, 2)).shape == (2, 2)


# ══════════════════════════════════════════════════════════════════
# eval_locate 的 --baseline / --save-baseline 卡口
# ══════════════════════════════════════════════════════════════════


def _fake_report(fitness: float):
    """构造一个 fitness 恰好等于给定值的 Report。

    ``Report.fitness`` 是算出来的（0.5*recall30 + 0.25*recall50 +
    0.25*(1-center_err)），所以这里反解 recall30、把另两项置为常量。
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from eval_locate import Report

    recall30 = (fitness - 0.25) / 0.5
    assert 0.0 <= recall30 <= 1.0, "测试用的 fitness 需落在 0.25~0.75 区间"
    return Report(
        n_images=6,
        n_gt=6,
        n_pred=6,
        tp=6,
        precision=0.9,
        recall=0.8,
        f1=0.85,
        mean_iou=0.7,
        center_err=0.0,
        top1_iou=0.8,
        top1_hit=0.9,
        recall30=recall30,
        recall50=0.0,
        recall70=0.0,
        center_hit=0.9,
        seconds=0.1,
    )


def _write_baseline(path: Path, fitness: float, n_samples: int = 6) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": "x",
                "n_samples": n_samples,
                "iou_threshold": 0.5,
                "metrics": {
                    "fitness": fitness,
                    "recall30": 0.5,
                    "recall50": 0.5,
                    "mean_iou": 0.7,
                    "precision": 0.9,
                    "center_hit": 0.9,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _namespace(baseline_path: Path, tolerance: float = 0.01):
    from argparse import Namespace

    return Namespace(baseline=str(baseline_path), tolerance=tolerance)


def test_check_baseline_passes_when_metrics_hold(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    from eval_locate import _check_against_baseline

    bp = tmp_path / "base.json"
    _write_baseline(bp, 0.60)
    # 现在 0.605，涨了 → 通过
    assert _check_against_baseline(_namespace(bp), _fake_report(0.605), 6) == 0


def test_check_baseline_fails_on_regression(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    from eval_locate import _check_against_baseline

    bp = tmp_path / "base.json"
    _write_baseline(bp, 0.60)
    # 现在 0.55，掉了 0.05 > 容忍 0.01 → 非零退出
    assert _check_against_baseline(_namespace(bp), _fake_report(0.55), 6) == 1


def test_check_baseline_tolerance_is_respected(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    from eval_locate import _check_against_baseline

    bp = tmp_path / "base.json"
    _write_baseline(bp, 0.60)
    # 掉 0.005 < 容忍 0.01 → 通过
    assert _check_against_baseline(_namespace(bp, 0.01), _fake_report(0.595), 6) == 0


def test_check_baseline_rejects_different_sample_count(tmp_path):
    """样本数不同就不可比，必须报错而不是给出误导性的结论。"""
    sys.path.insert(0, str(ROOT / "tools"))
    from eval_locate import _check_against_baseline

    bp = tmp_path / "base.json"
    _write_baseline(bp, 0.60, n_samples=320)
    assert _check_against_baseline(_namespace(bp), _fake_report(0.60), 6) == 2


def test_check_baseline_missing_file(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    from eval_locate import _check_against_baseline

    assert _check_against_baseline(_namespace(tmp_path / "nope.json"), _fake_report(0.6), 6) == 1


def test_eval_locate_exposes_baseline_flags():
    """CLI 必须真的暴露这两个开关，否则 CI 接不进去。"""
    src = (ROOT / "tools" / "eval_locate.py").read_text(encoding="utf-8")
    assert '"--save-baseline"' in src
    assert '"--baseline"' in src
    assert '"--tolerance"' in src
