"""Integration tests for the Point 1 eval bridge script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common.io.json_io import read_json, write_json


def _load_eval_module():
    module_path = Path("scripts/run_point1_eval.py")
    spec = importlib.util.spec_from_file_location("run_point1_eval_script", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load scripts/run_point1_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_point1_eval_writes_official_predictions_and_summary(tmp_path: Path) -> None:
    """The eval helper should export official-style predictions and an internal summary."""
    module = _load_eval_module()
    baseline_output = tmp_path / "baseline.json"
    official_output = tmp_path / "official.json"
    summary_output = tmp_path / "summary.json"
    registry_path = tmp_path / "registry.json"

    write_json(
        baseline_output,
        [
            {
                "image_id": "rule1-1",
                "provider_name": "demo",
                "model_name": "demo-model",
                "mode": "direct",
                "raw_response_text": "{}",
                "parsed_output": {
                    "image_id": "rule1-1",
                    "predictions": [
                        {
                            "rule_id": 1,
                            "decision_state": "violation",
                            "target_bbox": [0.1, 0.2, 0.3, 0.4],
                            "supporting_evidence_ids": [],
                            "counter_evidence_ids": [],
                            "unknown_items": [],
                            "reason_slots": {},
                            "reason_text": "Worker without hard hat.",
                            "confidence": 0.9,
                        },
                        {
                            "rule_id": 2,
                            "decision_state": "no_violation",
                            "target_bbox": None,
                            "supporting_evidence_ids": [],
                            "counter_evidence_ids": [],
                            "unknown_items": [],
                            "reason_slots": {},
                            "reason_text": "",
                            "confidence": 0.9,
                        },
                        {
                            "rule_id": 3,
                            "decision_state": "no_violation",
                            "target_bbox": None,
                            "supporting_evidence_ids": [],
                            "counter_evidence_ids": [],
                            "unknown_items": [],
                            "reason_slots": {},
                            "reason_text": "",
                            "confidence": 0.9,
                        },
                        {
                            "rule_id": 4,
                            "decision_state": "no_violation",
                            "target_bbox": None,
                            "supporting_evidence_ids": [],
                            "counter_evidence_ids": [],
                            "unknown_items": [],
                            "reason_slots": {},
                            "reason_text": "",
                            "confidence": 0.9,
                        },
                    ],
                },
                "error_message": None,
            }
        ],
    )
    write_json(
        registry_path,
        {
            "demo": ["rule1-1"],
            "demo_clean": [],
            "demo_rule1": ["rule1-1"],
            "demo_rule2": [],
            "demo_rule3": [],
            "demo_rule4": [],
        },
    )

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_eval.py",
            "--baseline-output",
            str(baseline_output),
            "--official-output",
            str(official_output),
            "--registry",
            str(registry_path),
            "--subset-name",
            "demo",
            "--summary-output",
            str(summary_output),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert read_json(official_output) == [
        {
            "image_id": "rule1-1",
            "rule_1_violation": {
                "bounding_box": [[0.1, 0.2, 0.3, 0.4]],
                "reason": "Worker without hard hat.",
            },
            "rule_2_violation": None,
            "rule_3_violation": None,
            "rule_4_violation": None,
        }
    ]
    assert read_json(summary_output)["rule_metrics"]["1"]["tp"] == 1


def test_run_point1_eval_supports_dataset_backed_summary(tmp_path: Path) -> None:
    """The eval helper should summarize against parquet truth when no registry subset is used."""
    module = _load_eval_module()
    baseline_output = tmp_path / "baseline.json"
    official_output = tmp_path / "official.json"
    summary_output = tmp_path / "summary.json"
    dataset_path = tmp_path / "test.parquet"

    write_json(
        baseline_output,
        [
            {
                "image_id": "img-1",
                "provider_name": "demo",
                "model_name": "demo-model",
                "mode": "direct",
                "raw_response_text": "{}",
                "parsed_output": {
                    "image_id": "img-1",
                    "predictions": [
                        {
                            "rule_id": 1,
                            "decision_state": "violation",
                            "target_bbox": [0.1, 0.2, 0.3, 0.4],
                            "supporting_evidence_ids": [],
                            "counter_evidence_ids": [],
                            "unknown_items": [],
                            "reason_slots": {},
                            "reason_text": "Worker without hard hat.",
                            "confidence": 0.9,
                        },
                        {
                            "rule_id": 2,
                            "decision_state": "no_violation",
                            "target_bbox": None,
                            "supporting_evidence_ids": [],
                            "counter_evidence_ids": [],
                            "unknown_items": [],
                            "reason_slots": {},
                            "reason_text": "",
                            "confidence": 0.9,
                        },
                        {
                            "rule_id": 3,
                            "decision_state": "no_violation",
                            "target_bbox": None,
                            "supporting_evidence_ids": [],
                            "counter_evidence_ids": [],
                            "unknown_items": [],
                            "reason_slots": {},
                            "reason_text": "",
                            "confidence": 0.9,
                        },
                        {
                            "rule_id": 4,
                            "decision_state": "no_violation",
                            "target_bbox": None,
                            "supporting_evidence_ids": [],
                            "counter_evidence_ids": [],
                            "unknown_items": [],
                            "reason_slots": {},
                            "reason_text": "",
                            "confidence": 0.9,
                        },
                    ],
                },
                "error_message": None,
            }
        ],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": "img-1",
                    "image_caption": "caption",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r1"},
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                }
            ]
        ),
        dataset_path,
    )

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_eval.py",
            "--baseline-output",
            str(baseline_output),
            "--official-output",
            str(official_output),
            "--target-parquet",
            str(dataset_path),
            "--summary-output",
            str(summary_output),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert read_json(summary_output)["truth_source"] == "dataset"
    assert read_json(summary_output)["rule_metrics"]["1"]["tp"] == 1
