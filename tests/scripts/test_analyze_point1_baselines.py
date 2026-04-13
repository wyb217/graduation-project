"""Integration tests for the baseline comparison script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common.io.json_io import read_json, write_json


def _load_analysis_module():
    module_path = Path("scripts/analyze_point1_baselines.py")
    spec = importlib.util.spec_from_file_location("analyze_point1_baselines_script", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load scripts/analyze_point1_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_point1_baselines_supports_dataset_backed_full_test_summary(
    tmp_path: Path,
) -> None:
    """The comparison helper should summarize direct/five-shot outputs against parquet truth."""
    module = _load_analysis_module()
    direct_output = tmp_path / "direct.json"
    few_shot_output = tmp_path / "few-shot.json"
    comparison_output = tmp_path / "comparison.json"
    dataset_path = tmp_path / "test.parquet"

    write_json(
        direct_output,
        [
            {
                "image_id": "img-1",
                "parsed_output": {
                    "image_id": "img-1",
                    "predictions": [
                        {"rule_id": 1, "decision_state": "violation"},
                        {"rule_id": 2, "decision_state": "no_violation"},
                        {"rule_id": 3, "decision_state": "no_violation"},
                        {"rule_id": 4, "decision_state": "no_violation"},
                    ],
                },
            }
        ],
    )
    write_json(
        few_shot_output,
        [
            {
                "image_id": "img-1",
                "parsed_output": {
                    "image_id": "img-1",
                    "predictions": [
                        {"rule_id": 1, "decision_state": "violation"},
                        {"rule_id": 2, "decision_state": "no_violation"},
                        {"rule_id": 3, "decision_state": "no_violation"},
                        {"rule_id": 4, "decision_state": "no_violation"},
                    ],
                },
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
            "analyze_point1_baselines.py",
            "--direct-output",
            str(direct_output),
            "--few-shot-output",
            str(few_shot_output),
            "--target-parquet",
            str(dataset_path),
            "--output",
            str(comparison_output),
        ]
        module.main()
    finally:
        sys.argv = argv

    comparison = read_json(comparison_output)
    assert comparison["direct"]["truth_source"] == "dataset"
    assert comparison["five_shot"]["rule_metrics"]["1"]["tp"] == 1
    assert "| rule1 |" in comparison["direct"]["rule_table_markdown"]
