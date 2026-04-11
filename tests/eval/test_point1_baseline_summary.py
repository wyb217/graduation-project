"""Tests for Point 1 baseline result summaries."""

from __future__ import annotations

import json
from pathlib import Path

from common.io.json_io import write_json
from eval.reports.point1_baseline_summary import summarize_baseline_run


def test_summarize_baseline_run_computes_parse_and_rule_metrics(tmp_path: Path) -> None:
    """The summary helper should compute parse success, bucket hits, and rule F1 values."""
    registry_path = tmp_path / "registry.json"
    write_json(
        registry_path,
        {
            "demo": ["clean-1", "rule1-1", "rule2-1", "rule3-1", "rule4-1"],
            "demo_clean": ["clean-1"],
            "demo_rule1": ["rule1-1"],
            "demo_rule2": ["rule2-1"],
            "demo_rule3": ["rule3-1"],
            "demo_rule4": ["rule4-1"],
        },
    )
    output_path = tmp_path / "baseline.json"
    output = [
        {
            "image_id": "clean-1",
            "parsed_output": {
                "image_id": "clean-1",
                "predictions": [
                    {"rule_id": 1, "decision_state": "no_violation"},
                    {"rule_id": 2, "decision_state": "no_violation"},
                    {"rule_id": 3, "decision_state": "no_violation"},
                    {"rule_id": 4, "decision_state": "no_violation"},
                ],
            },
        },
        {
            "image_id": "rule1-1",
            "parsed_output": {
                "image_id": "rule1-1",
                "predictions": [
                    {"rule_id": 1, "decision_state": "violation"},
                    {"rule_id": 2, "decision_state": "no_violation"},
                    {"rule_id": 3, "decision_state": "no_violation"},
                    {"rule_id": 4, "decision_state": "no_violation"},
                ],
            },
        },
        {"image_id": "rule2-1", "parsed_output": None},
        {
            "image_id": "rule3-1",
            "parsed_output": {
                "image_id": "rule3-1",
                "predictions": [
                    {"rule_id": 1, "decision_state": "no_violation"},
                    {"rule_id": 2, "decision_state": "no_violation"},
                    {"rule_id": 3, "decision_state": "violation"},
                    {"rule_id": 4, "decision_state": "no_violation"},
                ],
            },
        },
        {
            "image_id": "rule4-1",
            "parsed_output": {
                "image_id": "rule4-1",
                "predictions": [
                    {"rule_id": 1, "decision_state": "no_violation"},
                    {"rule_id": 2, "decision_state": "violation"},
                    {"rule_id": 3, "decision_state": "no_violation"},
                    {"rule_id": 4, "decision_state": "no_violation"},
                ],
            },
        },
    ]
    output_path.write_text(json.dumps(output), encoding="utf-8")

    summary = summarize_baseline_run(
        output_path=output_path,
        registry_path=registry_path,
        subset_name="demo",
    )

    assert summary["num_records"] == 5
    assert summary["num_parsed"] == 4
    assert summary["parse_success_rate"] == 0.8
    assert summary["bucket_hit_rate"]["clean"]["correct"] == 1
    assert summary["bucket_hit_rate"]["rule1"]["correct"] == 1
    assert summary["bucket_hit_rate"]["rule2"]["correct"] == 0
    assert summary["rule_metrics"]["1"]["tp"] == 1
    assert summary["rule_metrics"]["2"]["fp"] == 1
