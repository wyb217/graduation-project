"""Tests for the Rule 1 small-loop summary helper."""

from __future__ import annotations

from pathlib import Path

from common.io.json_io import write_json
from eval.reports.point1_rule1_summary import (
    summarize_rule1_bucketed_run,
    summarize_rule1_smallloop,
)


def test_summarize_rule1_smallloop_reports_binary_metrics_and_unknown_rates(
    tmp_path: Path,
) -> None:
    """Rule 1 summaries should surface precision/recall and abstention behavior."""
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"

    write_json(
        registry_path,
        {
            "demo_clean": ["clean-1", "clean-2"],
            "demo_rule1": ["rule1-1", "rule1-2"],
        },
    )
    write_json(
        output_path,
        [
            {
                "image_id": "clean-1",
                "parsed_output": {
                    "image_id": "clean-1",
                    "predictions": [{"rule_id": 1, "decision_state": "no_violation"}],
                },
            },
            {
                "image_id": "clean-2",
                "parsed_output": {
                    "image_id": "clean-2",
                    "predictions": [{"rule_id": 1, "decision_state": "unknown"}],
                },
            },
            {
                "image_id": "rule1-1",
                "parsed_output": {
                    "image_id": "rule1-1",
                    "predictions": [{"rule_id": 1, "decision_state": "violation"}],
                },
            },
            {
                "image_id": "rule1-2",
                "parsed_output": {
                    "image_id": "rule1-2",
                    "predictions": [{"rule_id": 1, "decision_state": "unknown"}],
                },
            },
        ],
    )

    summary = summarize_rule1_smallloop(
        output_path=output_path,
        registry_path=registry_path,
        clean_split_name="demo_clean",
        positive_split_name="demo_rule1",
    )

    assert summary["num_records"] == 4
    assert summary["num_parsed"] == 4
    assert summary["parse_success_rate"] == 1.0
    assert summary["rule1_precision"] == 1.0
    assert summary["rule1_recall"] == 0.5
    assert round(summary["rule1_f1"], 4) == 0.6667
    assert summary["clean_hit_rate"] == 0.5
    assert summary["rule1_hit_rate"] == 0.5
    assert summary["unknown_rate_overall"] == 0.5
    assert summary["unknown_rate_clean"] == 0.5
    assert summary["unknown_rate_rule1"] == 0.5


def test_summarize_rule1_bucketed_run_reports_false_positives_by_bucket(tmp_path: Path) -> None:
    """Bucketed Rule 1 summaries should expose where cross-rule false positives come from."""
    output_path = tmp_path / "rule1.json"
    write_json(
        output_path,
        [
            {
                "image_id": "clean-1",
                "parsed_output": {
                    "image_id": "clean-1",
                    "predictions": [{"rule_id": 1, "decision_state": "violation"}],
                },
            },
            {
                "image_id": "rule2-1",
                "parsed_output": {
                    "image_id": "rule2-1",
                    "predictions": [{"rule_id": 1, "decision_state": "unknown"}],
                },
            },
            {
                "image_id": "rule4-1",
                "parsed_output": {
                    "image_id": "rule4-1",
                    "predictions": [{"rule_id": 1, "decision_state": "violation"}],
                },
            },
            {
                "image_id": "rule1-1",
                "parsed_output": {
                    "image_id": "rule1-1",
                    "predictions": [{"rule_id": 1, "decision_state": "violation"}],
                },
            },
        ],
    )

    summary = summarize_rule1_bucketed_run(
        output_path=output_path,
        bucket_image_ids={
            "clean": ("clean-1",),
            "rule2": ("rule2-1",),
            "rule4": ("rule4-1",),
            "rule1": ("rule1-1",),
        },
        positive_bucket_name="rule1",
    )

    assert summary["rule1_tp"] == 1
    assert summary["rule1_fp"] == 2
    assert summary["fp_by_bucket"] == {"clean": 1, "rule2": 0, "rule4": 1}
    assert summary["negative_hit_rate"] == 0.0
    assert summary["bucket_hit_rate"]["rule1"]["correct"] == 1
