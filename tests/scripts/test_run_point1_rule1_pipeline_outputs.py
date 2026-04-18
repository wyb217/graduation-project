"""Summary and output-shape tests for the Rule 1 runner script."""

from __future__ import annotations

import sys
from pathlib import Path

from common.io.json_io import read_json
from common.schemas.bbox import NormalizedBBox
from common.schemas.point1 import Point1BaselineRecord, Point1ImagePredictionSet, Point1Prediction
from point1.pipelines import rule1_runner_data as data_module
from tests.scripts.rule1_pipeline_script_helper import (
    build_base_row,
    load_rule1_script_module,
    write_registry_payload,
    write_rule1_rows,
)


def test_run_point1_rule1_pipeline_supports_fulltest_mode_without_registry(
    tmp_path: Path,
) -> None:
    """The script should support dataset-backed Rule 1 fulltest summaries."""
    module = load_rule1_script_module()
    dataset_path = tmp_path / "target.parquet"
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

    built: dict[str, object] = {}

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        built["num_samples"] = len(target_samples)
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

    def fake_summarize_rule1_run_from_dataset(**kwargs):  # noqa: ANN003
        built["summary_kwargs"] = kwargs
        return {"rule1_precision": 1.0, "rule1_recall": 1.0}

    module.run_rule1_pipeline = fake_run_rule1_pipeline
    data_module.summarize_rule1_run_from_dataset = fake_summarize_rule1_run_from_dataset

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_rule1_pipeline.py",
            "--fulltest",
            "--target-parquet",
            str(dataset_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert built["num_samples"] == 2
    assert built["summary_kwargs"] == {
        "output_path": output_path,
        "target_parquet_paths": (dataset_path,),
    }
    assert read_json(summary_path)["rule1_precision"] == 1.0


def test_run_point1_rule1_pipeline_supports_bucketed_summary_with_explicit_positive_split(
    tmp_path: Path,
) -> None:
    """The script should support 65-style bucketed summaries for later BML runs."""
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
                image_id="rule2-1",
                image_bytes=b"fake-image-rule2",
                image_path="rule2.jpg",
                rule_2_violation={"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r2"},
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
            "demo_rule2": ["rule2-1"],
            "demo_rule1": ["rule1-1"],
        },
    )

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
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
                            decision_state="violation",
                            target_bbox=NormalizedBBox(0.1, 0.2, 0.3, 0.4),
                            supporting_evidence_ids=("person-1:hard_hat_visible",),
                            counter_evidence_ids=(),
                            unknown_items=(),
                            reason_slots={},
                            reason_text="clean-fp",
                            confidence=0.9,
                        ),
                    ),
                ),
            ),
            Point1BaselineRecord(
                image_id="rule2-1",
                provider_name="rule1_pipeline",
                model_name="opencv_hog+heuristic_rule1",
                mode="rule1_smallloop",
                raw_response_text="",
                parsed_output=Point1ImagePredictionSet(
                    image_id="rule2-1",
                    predictions=(
                        Point1Prediction(
                            rule_id=1,
                            decision_state="unknown",
                            target_bbox=None,
                            supporting_evidence_ids=(),
                            counter_evidence_ids=(),
                            unknown_items=("ppe_applicable",),
                            reason_slots={},
                            reason_text="unknown",
                            confidence=0.2,
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
                            reason_text="tp",
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
            "demo_rule2",
            "demo_rule1",
            "--positive-split-name",
            "demo_rule1",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    summary = read_json(summary_path)
    assert summary["fp_by_bucket"] == {"demo_clean": 1, "demo_rule2": 0}
    assert summary["negative_hit_rate"] == 0.0


def test_run_point1_rule1_pipeline_supports_progress_checkpoint_and_failure_outputs(
    tmp_path: Path,
) -> None:
    """The script should wire optional observability outputs through the runner."""
    module = load_rule1_script_module()
    dataset_path = tmp_path / "target.parquet"
    output_path = tmp_path / "rule1.json"
    summary_path = tmp_path / "rule1.summary.json"
    progress_path = tmp_path / "rule1.progress.json"
    checkpoint_path = tmp_path / "rule1.checkpoint.json"
    failure_path = tmp_path / "rule1.failures.json"

    write_rule1_rows(
        dataset_path,
        [
            build_base_row(
                image_id="clean-1",
                image_bytes=b"fake-image-clean",
                image_path="clean.jpg",
            )
        ],
    )

    built: dict[str, object] = {}

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        built["run_kwargs"] = kwargs
        return []

    def fake_export_rule1_failures(**kwargs):  # noqa: ANN003
        built["failure_kwargs"] = kwargs
        return {
            "counts": {
                "false_positives": 0,
                "false_negatives": 0,
                "unknown_predictions": 0,
                "parse_failures": 0,
            },
            "records": [],
        }

    module.run_rule1_pipeline = fake_run_rule1_pipeline
    module.export_rule1_failures = fake_export_rule1_failures

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_rule1_pipeline.py",
            "--fulltest",
            "--target-parquet",
            str(dataset_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--progress-output",
            str(progress_path),
            "--checkpoint-output",
            str(checkpoint_path),
            "--checkpoint-every",
            "5",
            "--failure-output",
            str(failure_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert built["run_kwargs"]["progress_output"] == progress_path
    assert built["run_kwargs"]["checkpoint_output"] == checkpoint_path
    assert built["run_kwargs"]["checkpoint_every"] == 5
    assert built["failure_kwargs"]["output_path"] == output_path
    assert failure_path.exists()


def test_build_rule1_summary_includes_runtime_timing_statistics(
    tmp_path: Path,
) -> None:
    """Runner summaries should include aggregate timing evidence from output records."""
    output_path = tmp_path / "rule1.json"
    target_path = tmp_path / "target.parquet"
    output_path.write_text(
        """
        [
          {
            "image_id": "a",
            "provider_name": "rule1_pipeline_local_qwen",
            "model_name": "demo",
            "mode": "rule1_smallloop_local_qwen",
            "raw_response_text": "",
            "parsed_output": null,
            "error_message": null,
            "candidate_ms": 10.0,
            "predicate_ms": 50.0,
            "executor_ms": 5.0,
            "total_ms": 65.0,
            "candidate_count": 1,
            "fallback_used": false,
            "predicate_backend": "local_qwen",
            "candidate_batch_size": 1,
            "max_new_tokens": 256
          },
          {
            "image_id": "b",
            "provider_name": "rule1_pipeline_local_qwen",
            "model_name": "demo",
            "mode": "rule1_smallloop_local_qwen",
            "raw_response_text": "",
            "parsed_output": null,
            "error_message": null,
            "candidate_ms": 20.0,
            "predicate_ms": 70.0,
            "executor_ms": 15.0,
            "total_ms": 105.0,
            "candidate_count": 2,
            "fallback_used": true,
            "predicate_backend": "local_qwen",
            "candidate_batch_size": 1,
            "max_new_tokens": 256
          }
        ]
        """,
        encoding="utf-8",
    )

    args = type(
        "Args",
        (),
        {
            "target_parquet": [target_path],
            "limit": None,
            "target_split_names": None,
            "registry": None,
        },
    )()

    original = data_module.summarize_rule1_run_from_dataset
    try:
        data_module.summarize_rule1_run_from_dataset = lambda **kwargs: {"rule1_precision": 1.0}
        summary = data_module.build_rule1_summary(
            args=args,
            output_path=output_path,
            target_samples=(),
            summary_context={"mode": "fulltest"},
        )
    finally:
        data_module.summarize_rule1_run_from_dataset = original

    assert summary["rule1_precision"] == 1.0
    assert summary["timing_ms"]["predicate"]["mean"] == 60.0
    assert summary["timing_ms"]["predicate"]["p50"] == 50.0
    assert summary["timing_ms"]["predicate"]["p95"] == 70.0
    assert summary["timing_ms"]["total"]["mean"] == 85.0
    assert summary["candidate_count_stats"]["mean"] == 1.5
    assert summary["fallback_rate"] == 0.5
    assert summary["runtime_config"] == {
        "predicate_backend": "local_qwen",
        "candidate_batch_size": 1,
        "max_new_tokens": 256,
    }
