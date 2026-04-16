"""Basic execution tests for the Rule 1 runner script."""

from __future__ import annotations

import sys
from pathlib import Path

from common.io.json_io import read_json
from common.schemas.bbox import NormalizedBBox
from common.schemas.point1 import Point1BaselineRecord, Point1ImagePredictionSet, Point1Prediction
from tests.scripts.rule1_pipeline_script_helper import (
    build_base_row,
    load_rule1_script_module,
    write_registry_payload,
    write_rule1_rows,
)


def test_run_point1_rule1_pipeline_writes_output_and_summary(tmp_path: Path) -> None:
    """The Rule 1 script should load a frozen subset and emit a binary summary."""
    module = load_rule1_script_module()
    dataset_path = tmp_path / "target.parquet"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"
    summary_path = tmp_path / "rule1.summary.json"

    write_rule1_rows(
        dataset_path,
        [
            build_base_row(
                image_id="clean-1",
                image_bytes=b"fake-image-clean",
                image_path="clean.jpg",
            ),
            build_base_row(
                image_id="rule1-1",
                image_bytes=b"fake-image-rule1",
                image_path="rule1.jpg",
                rule_1_violation={"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r1"},
            ),
        ],
    )
    write_registry_payload(
        registry_path,
        {
            "demo_clean": ["clean-1"],
            "demo_rule1": ["rule1-1"],
        },
    )

    calls: dict[str, object] = {}

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        calls["num_samples"] = len(target_samples)
        calls["has_bytes"] = all(
            sample.image is not None and sample.image.bytes is not None for sample in target_samples
        )
        calls["extra_kwargs"] = kwargs
        return [
            Point1BaselineRecord(
                image_id="clean-1",
                provider_name="rule1_pipeline",
                model_name="opencv_hog+heuristic_rule1",
                mode="rule1_smallloop",
                raw_response_text="",
                parsed_output=Point1ImagePredictionSet(
                    image_id="clean-1",
                    predictions=(
                        Point1Prediction(
                            rule_id=1,
                            decision_state="no_violation",
                            target_bbox=None,
                            supporting_evidence_ids=(),
                            counter_evidence_ids=("person-1:hard_hat_visible",),
                            unknown_items=(),
                            reason_slots={},
                            reason_text="clean",
                            confidence=0.9,
                        ),
                    ),
                ),
            ),
            Point1BaselineRecord(
                image_id="rule1-1",
                provider_name="rule1_pipeline",
                model_name="opencv_hog+heuristic_rule1",
                mode="rule1_smallloop",
                raw_response_text="",
                parsed_output=Point1ImagePredictionSet(
                    image_id="rule1-1",
                    predictions=(
                        Point1Prediction(
                            rule_id=1,
                            decision_state="violation",
                            target_bbox=NormalizedBBox(0.1, 0.2, 0.3, 0.4),
                            supporting_evidence_ids=("person-1:hard_hat_visible",),
                            counter_evidence_ids=(),
                            unknown_items=(),
                            reason_slots={},
                            reason_text="rule1",
                            confidence=0.9,
                        ),
                    ),
                ),
            ),
        ]

    module.run_rule1_pipeline = fake_run_rule1_pipeline

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_rule1_pipeline.py",
            "--target-parquet",
            str(dataset_path),
            "--registry",
            str(registry_path),
            "--target-split-names",
            "demo_clean",
            "demo_rule1",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert calls["num_samples"] == 2
    assert calls["has_bytes"] is True
    assert calls["extra_kwargs"]["provider_name"] == "rule1_pipeline"
    assert len(read_json(output_path)) == 2
    summary = read_json(summary_path)
    assert summary["rule1_precision"] == 1.0
    assert summary["rule1_recall"] == 1.0
