"""Tests for the Rule 1 small-loop summary helper."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common.io.json_io import write_json
from eval.reports.point1_rule1_summary import (
    summarize_rule1_bucketed_run,
    summarize_rule1_run_from_dataset,
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


def test_summarize_rule1_run_from_dataset_reports_fulltest_metrics(tmp_path: Path) -> None:
    """Dataset-backed Rule 1 summaries should work without a frozen subset registry."""
    dataset_path = tmp_path / "target.parquet"
    output_path = tmp_path / "rule1.json"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": "clean-1",
                    "image": {"bytes": b"img-a", "path": "a.jpg"},
                    "image_caption": "clean sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": None,
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
                {
                    "image_id": "rule1-1",
                    "image": {"bytes": b"img-b", "path": "b.jpg"},
                    "image_caption": "rule1 sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r1"},
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
                {
                    "image_id": "rule1-2",
                    "image": {"bytes": b"img-c", "path": "c.jpg"},
                    "image_caption": "rule1 sample 2",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": {"bounding_box": [[0.2, 0.3, 0.4, 0.5]], "reason": "r1"},
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
            ]
        ),
        dataset_path,
    )

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

    summary = summarize_rule1_run_from_dataset(
        output_path=output_path,
        target_parquet_paths=(dataset_path,),
    )

    assert summary["num_records"] == 3
    assert summary["num_parsed"] == 3
    assert summary["parse_success_rate"] == 1.0
    assert summary["rule1_precision"] == 0.5
    assert summary["rule1_recall"] == 0.5
    assert round(summary["rule1_f1"], 4) == 0.5
    assert summary["rule1_tp"] == 1
    assert summary["rule1_fp"] == 1
    assert summary["rule1_fn"] == 1
    assert summary["positive_support"] == 2
    assert summary["negative_support"] == 1
    assert summary["state_counts"]["unknown"] == 1
