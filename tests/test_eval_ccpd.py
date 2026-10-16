"""``tools/eval_ccpd.py`` 与 ``tools/eval_report.py`` 的测试。

评测工具不参与推理，但它产出的数字会被引用到文档和结论里 —— 算错了
比推理错更危险，因为看不出来。所以这里把三件容易悄悄漂移的事钉死：

* **``--engines`` 规格解析**：``fusion:chain`` 与 ``--engine fusion
  --fusion-strategy chain`` 必须等价，否则文档里的"复现命令"假的；
* **编辑距离回溯**（``_diff_pairs``）：形近字统计全靠它，回溯写错会
  把替换报成插入/删除，混淆对就全错；
* **失败主类的边界**：``无框`` 与 ``有框无字`` 只差一个 ``best_iou``
  判断，混淆了会把"定位失败"和"识别失败"统计成一个数，从而给出
  错误的改进方向。

``eval_report.py`` 的表按 :data:`eval_ccpd.FAIL_KEYS` 取数，因此额外
断言两张表的列数与常量一致 —— 这是两文件之间唯一的隐式契约。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import eval_report  # noqa: E402
from eval_ccpd import (  # noqa: E402
    FAIL_KEYS,
    FAIL_LABELS,
    _diff_pairs,
    _edit_distance,
    classify_failures,
    parse_spec,
    taxonomy_to_json,
)


def make_item(
    name: str,
    gt: str,
    pred: list[str],
    *,
    best_iou: float = 0.9,
    subset: str = "base",
) -> dict[str, Any]:
    """造一条 ``eval_ccpd`` 的逐图明细（``items`` 里的元素）。"""
    return {
        "name": name,
        "subset": subset,
        "gt": gt,
        "pred": pred,
        "exact": gt in pred,
        "near": False,
        "extra": 0,
        "char_acc": 1.0 if gt in pred else 0.0,
        "best_iou": best_iou,
        "pred_box": None,
        "gt_box": None,
        "ms": 1.0,
        "fusion": {},
    }


class TestParseSpec:
    """``--engines cv,deep,fusion:chain`` 的规格解析。"""

    def test_bare_name_has_empty_strategy(self):
        assert parse_spec("deep") == ("deep", "")

    def test_colon_carries_strategy(self):
        assert parse_spec("fusion:chain") == ("fusion", "chain")

    def test_whitespace_is_stripped(self):
        assert parse_spec(" fusion : auto ") == ("fusion", "auto")

    def test_colon_without_strategy_keeps_empty(self):
        assert parse_spec("fusion:") == ("fusion", "")

    def test_unknown_engine_name_is_passed_through_for_caller_to_reject(self):
        # 解析不做白名单校验 —— 由 main() 统一报错，解析器保持纯粹
        assert parse_spec("nope:x") == ("nope", "x")


class TestFailKeyContract:
    """分类键是脚本间的契约，不能随手改。"""

    def test_keys_are_ascii_and_stable(self):
        assert FAIL_KEYS == ("no_box", "no_text", "near", "gross")

    def test_every_key_has_a_label(self):
        assert set(FAIL_LABELS) == set(FAIL_KEYS)

    def test_labels_are_chinese_and_nonempty(self):
        for key in FAIL_KEYS:
            assert FAIL_LABELS[key].strip()


class TestDiffPairs:
    """编辑距离回溯：形近字统计的基础。"""

    def test_single_substitution_is_reported(self):
        assert _diff_pairs("皖AFZ768", "皖AF2768") == [("Z", "2")]

    def test_identical_strings_yield_no_pairs(self):
        assert _diff_pairs("皖A12345", "皖A12345") == []

    def test_trailing_deletion(self):
        assert _diff_pairs("皖AR786A", "皖AR786") == [("A", "<del>")]

    def test_missing_character_is_marked_as_deletion(self):
        assert _diff_pairs("皖ASS011", "皖ASS0") == [("1", "<del>"), ("1", "<del>")]

    def test_extra_character_is_marked_as_insertion(self):
        assert _diff_pairs("皖A03728", "皖A0372X8") == [("<ins>", "X")]

    def test_province_confusion_pair(self):
        assert _diff_pairs("赣AF6022", "皖AF6022") == [("赣", "皖")]

    def test_pairs_are_ordered_left_to_right(self):
        pairs = _diff_pairs("Z0A", "27A")
        assert pairs == [("Z", "2"), ("0", "7")]

    def test_pair_count_matches_edit_distance(self):
        for a, b in [("皖AFZ768", "皖AF2768"), ("皖A8W727", "皖B8722"), ("赣A1", "皖B199")]:
            assert len(_diff_pairs(a, b)) == _edit_distance(a, b)


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("皖A1234", "皖A1234") == 0

    def test_one_substitution(self):
        assert _edit_distance("皖AL813D", "皖AL8130") == 1

    def test_empty_side(self):
        assert _edit_distance("", "皖A") == 2

    def test_symmetric(self):
        assert _edit_distance("皖A123", "皖A1X3") == _edit_distance("皖A1X3", "皖A123")


class TestClassifyFailures:
    """四类互斥，按"卡在哪一环"分 —— 混淆了就会开错药方。"""

    def test_no_box_when_nothing_detected(self):
        payload = {"items": [make_item("a.jpg", "皖A12345", [], best_iou=0.0)]}
        tx = classify_failures(payload, 0.5)
        assert [x["name"] for x in tx["cats"]["no_box"]] == ["a.jpg"]
        assert tx["cats"]["no_text"] == []

    def test_no_text_when_box_is_right_but_nothing_read(self):
        payload = {"items": [make_item("a.jpg", "皖A12345", [], best_iou=0.91)]}
        tx = classify_failures(payload, 0.5)
        assert [x["name"] for x in tx["cats"]["no_text"]] == ["a.jpg"]
        assert tx["cats"]["no_box"] == []

    def test_near_miss_is_one_char_apart(self):
        payload = {"items": [make_item("a.jpg", "皖AFZ768", ["皖AF2768"])]}
        tx = classify_failures(payload, 0.5)
        assert [x["name"] for x in tx["cats"]["near"]] == ["a.jpg"]
        assert tx["pairs"] == [(("Z", "2"), 1)]

    def test_gross_error_is_two_or_more_chars_apart(self):
        payload = {"items": [make_item("a.jpg", "皖A8W727", ["皖B8722"])]}
        tx = classify_failures(payload, 0.5)
        assert [x["name"] for x in tx["cats"]["gross"]] == ["a.jpg"]
        assert tx["pairs"] == []

    def test_exact_hit_stays_out_of_every_class(self):
        payload = {"items": [make_item("a.jpg", "皖A12345", ["皖A12345"])]}
        tx = classify_failures(payload, 0.5)
        assert tx["total_fail"] == 0
        assert all(not rows for rows in tx["cats"].values())

    def test_classes_are_mutually_exclusive_and_cover_all_failures(self):
        payload = {
            "items": [
                make_item("hit.jpg", "皖A12345", ["皖A12345"]),
                make_item("box.jpg", "皖A12345", [], best_iou=0.0),
                make_item("text.jpg", "皖A12345", [], best_iou=0.8),
                make_item("near.jpg", "皖AFZ768", ["皖AF2768"]),
                make_item("gross.jpg", "皖A8W727", ["皖B8722"]),
            ]
        }
        tx = classify_failures(payload, 0.5)
        assert tx["total_fail"] == 4
        assert sum(len(rows) for rows in tx["cats"].values()) == 4

    def test_best_of_several_predictions_decides_the_class(self):
        # 多余的误检不该把"差一个字符"升级成"严重错读"
        payload = {"items": [make_item("a.jpg", "皖AFZ768", ["琼A11111", "皖AF2768", "川J1111T"])]}
        tx = classify_failures(payload, 0.5)
        assert [x["name"] for x in tx["cats"]["near"]] == ["a.jpg"]

    def test_loc_miss_is_orthogonal_to_the_four_classes(self):
        # 形近字，但框也歪了 —— 两个统计都要记它
        payload = {"items": [make_item("a.jpg", "皖AFZ768", ["皖AF2768"], best_iou=0.2)]}
        tx = classify_failures(payload, 0.5)
        assert [x["name"] for x in tx["cats"]["near"]] == ["a.jpg"]
        assert [x["name"] for x in tx["loc_miss"]] == ["a.jpg"]

    def test_iou_threshold_is_honoured_at_the_boundary(self):
        payload = {"items": [make_item("a.jpg", "皖A12345", [], best_iou=0.5)]}
        assert classify_failures(payload, 0.5)["cats"]["no_text"]
        assert classify_failures(payload, 0.5001)["cats"]["no_box"]

    def test_by_subset_counts_totals_and_failures(self):
        payload = {
            "items": [
                make_item("a.jpg", "皖A12345", ["皖A12345"], subset="base"),
                make_item("b.jpg", "皖A12345", [], best_iou=0.0, subset="base"),
                make_item("c.jpg", "皖AFZ768", ["皖AF2768"], subset="blur"),
            ]
        }
        by_subset = classify_failures(payload, 0.5)["by_subset"]
        assert by_subset["base"] == {"total": 2, "fail": 1}
        assert by_subset["blur"] == {"total": 1, "fail": 1}

    def test_confusion_pairs_are_ranked_by_frequency(self):
        payload = {
            "items": [
                make_item("a.jpg", "皖AFZ768", ["皖AF2768"]),  # Z→2
                make_item("b.jpg", "皖A3Z682", ["皖A32682"]),  # Z→2（同一对，应累计）
                make_item("c.jpg", "皖AL813D", ["皖AL8130"]),  # D→0
            ]
        }
        pairs = classify_failures(payload, 0.5)["pairs"]
        assert pairs[0] == (("Z", "2"), 2)
        assert (("D", "0"), 1) in pairs

    def test_z_seven_confusion_is_not_folded_into_z_two(self):
        """真实数据里 ``皖AU3Z28 → 皖AU3728`` 混的是 Z→7；别和 Z→2 合成一个数。"""
        payload = {"items": [make_item("a.jpg", "皖AU3Z28", ["皖AU3728"])]}
        pairs = classify_failures(payload, 0.5)["pairs"]
        assert pairs == [(("Z", "7"), 1)]


class TestTaxonomyToJson:
    def test_keeps_only_names_not_full_details(self):
        payload = {"items": [make_item("a.jpg", "皖AFZ768", ["皖AF2768"])]}
        blob = taxonomy_to_json(classify_failures(payload, 0.5))
        assert blob["cats"]["near"] == ["a.jpg"]
        assert blob["total_fail"] == 1
        assert blob["iou_threshold"] == 0.5

    def test_is_json_serialisable(self):
        import json

        payload = {"items": [make_item("a.jpg", "皖AFZ768", ["皖AF2768"])]}
        blob = taxonomy_to_json(classify_failures(payload, 0.5))
        assert json.loads(json.dumps(blob, ensure_ascii=False))["cats"]["near"] == ["a.jpg"]

    def test_all_keys_present_even_when_empty(self):
        payload = {"items": [make_item("a.jpg", "皖A12345", ["皖A12345"])]}
        blob = taxonomy_to_json(classify_failures(payload, 0.5))
        assert set(blob["cats"]) == set(FAIL_KEYS)


def make_report(out_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """造一份最小的 ``eval_ccpd --json`` 产物。"""
    items = [
        make_item("a.jpg", "皖A12345", ["皖A12345"], subset="base"),
        make_item("b.jpg", "皖AFZ768", ["皖AF2768"], subset="blur"),
    ]
    payload = {
        "engine": "deep",
        "summary": {
            "n": 2,
            "img_hit": 1,
            "img_hit_rate": 0.5,
            "char_acc": 0.875,
            "near": 1,
            "extra": 0,
            "loc_hit": 1,
            "loc_hit_rate": 0.5,
            "loc_precision": 0.5,
            "det_total": 2,
            "det_tp": 1,
            "mean_best_iou": 0.7,
            "ms_median": 10.0,
            "ms_p95": 20.0,
            "loc_f1": 0.5,
            "per_subset": {"base": [1, 1, 1], "blur": [1, 0, 0]},
        },
        "items": items,
        "fails": [],
        "taxonomy": taxonomy_to_json(classify_failures({"items": items}, 0.5)),
    }
    return {"deep": payload, **(out_overrides or {})}


class TestEvalReportTables:
    """表列数必须与 FAIL_KEYS 常量一致 —— 两文件之间唯一的隐式契约。"""

    def test_overview_has_one_row_per_config(self):
        table = eval_report.table_overview(make_report())
        assert table.count("\n") == 2  # 表头 + 分隔 + 1 行数据
        assert "| deep |" in table

    def test_overview_handles_missing_summary(self):
        table = eval_report.table_overview({"x": {"engine": "x"}})
        assert "| x |" in table

    def test_taxonomy_table_column_count_matches_fail_keys(self):
        table = eval_report.table_taxonomy(make_report())
        header = table.splitlines()[0]
        # 配置 + 未命中 + len(FAIL_KEYS) + 定位失手 + 混淆对
        assert len(header.split("|")) - 2 == len(FAIL_KEYS) + 4

    def test_taxonomy_table_header_bridge_and_rows_are_same_width(self):
        """三处不同宽 → markdown 表静默错位，且渲染出来看不出是代码的问题。"""
        table = eval_report.table_taxonomy(make_report())
        widths = {len(line.split("|")) for line in table.splitlines()}
        assert len(widths) == 1, f"列数不一致: {widths}"

    def test_taxonomy_table_survives_a_report_without_taxonomy(self):
        table = eval_report.table_taxonomy({"cv": {"engine": "cv"}})
        assert "| cv | — |" in table
        # 缺 taxonomy 时列数仍需对齐，否则 markdown 表会错位
        assert len({len(x.split("|")) for x in table.splitlines()}) == 1

    def test_cell_counts_match_taxonomy_lengths(self):
        report = make_report()
        tx = report["deep"]["taxonomy"]
        row = eval_report.table_taxonomy(report).splitlines()[2]
        assert f"| {tx['total_fail']} |" in row
        assert f"| {len(tx['cats']['near'])} |" in row

    def test_subsets_table_lists_every_subset_present(self):
        table = eval_report.table_subsets(make_report())
        assert "base" in table
        assert "blur" in table
        assert "**合计**" in table

    def test_complementarity_counts_engine_exclusive_hits(self):
        payload = make_report()["deep"]
        cv_same = {**payload, "engine": "cv"}
        table = eval_report.table_complementarity({"deep": payload, "cv": cv_same})
        # 两个引擎命中集合相同 → 互为独占 0，各自 1 张（自身）
        assert "| **deep** 独占 | 1（自身） | 0 |" in table
        assert "| **cv** 独占 | 0 | 1（自身） |" in table

    def test_complementarity_reports_the_router_upper_bound(self):
        payload = make_report()["deep"]
        table = eval_report.table_complementarity({"deep": payload})
        assert "至少一个命中" in table
        assert "1/2" in table

    def test_hard_samples_lists_only_images_every_config_misses(self):
        deep = make_report()["deep"]
        text = eval_report.list_hard({"deep": deep})
        assert "b.jpg" in text
        assert "a.jpg" not in text

    def test_main_writes_all_sections(self, tmp_path: Path, capsys):
        import json

        path = tmp_path / "eval.json"
        path.write_text(json.dumps(make_report(), ensure_ascii=False), encoding="utf-8")
        old_argv = sys.argv
        sys.argv = ["eval_report.py", str(path)]
        try:
            assert eval_report.main() == 0
        finally:
            sys.argv = old_argv
        out = capsys.readouterr().out
        for section in ("总览", "分子集", "失败分类", "互补性", "硬样本"):
            assert section in out

    def test_main_reports_missing_file(self, tmp_path: Path, capsys):
        old_argv = sys.argv
        sys.argv = ["eval_report.py", str(tmp_path / "nope.json")]
        try:
            assert eval_report.main() == 1
        finally:
            sys.argv = old_argv
        assert "找不到" in capsys.readouterr().out
