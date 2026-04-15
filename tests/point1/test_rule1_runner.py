"""Tests for the Rule 1 small-loop runner."""

from __future__ import annotations

from pathlib import Path

from benchmark.constructionsite10k.parser import parse_sample
from common.io.json_io import read_json
from common.schemas.bbox import NormalizedBBox
from common.schemas.point1 import Point1Prediction
from point1.pipelines.rule1 import Rule1PipelineResult
from point1.pipelines.runner import run_rule1_pipeline


class FakeRule1Pipeline:
    """Simple fake pipeline for runner tests."""

    def __init__(self, results_by_image_id: dict[str, Rule1PipelineResult | Exception]) -> None:
        self.results_by_image_id = results_by_image_id

    def run(self, sample) -> Rule1PipelineResult:  # noqa: ANN001
        result = self.results_by_image_id[sample.image_id]
        if isinstance(result, Exception):
            raise result
        return result


def _build_image_prediction(*, image_id: str, decision_state: str) -> Rule1PipelineResult:
    prediction = Point1Prediction(
        rule_id=1,
        decision_state=decision_state,
        target_bbox=(NormalizedBBox(0.1, 0.2, 0.3, 0.4) if decision_state == "violation" else None),
        supporting_evidence_ids=(
            ("person-1:hard_hat_visible",) if decision_state == "violation" else ()
        ),
        counter_evidence_ids=(
            () if decision_state == "violation" else ("person-1:hard_hat_visible",)
        ),
        unknown_items=("person_visible",) if decision_state == "unknown" else (),
        reason_slots={
            "subject": "worker candidate 1",
            "missing_item": "",
            "scene_condition": "site",
        },
        reason_text=f"{image_id}:{decision_state}",
        confidence=0.9,
    )
    return Rule1PipelineResult(
        image_id=image_id,
        candidate_predictions=(prediction,),
        image_prediction=prediction,
    )


def test_run_rule1_pipeline_wraps_image_level_predictions_as_baseline_records(
    sample_annotation: dict[str, object],
) -> None:
    """The runner should emit baseline-compatible image-level records."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"image-a", "path": "a.jpg"},
        }
    )
    records = run_rule1_pipeline(
        target_samples=(sample,),
        pipeline=FakeRule1Pipeline(
            {
                sample.image_id: _build_image_prediction(
                    image_id=sample.image_id,
                    decision_state="violation",
                )
            }
        ),
    )

    assert len(records) == 1
    assert records[0].provider_name == "rule1_pipeline"
    assert records[0].model_name == "opencv_hog+heuristic_rule1"
    assert records[0].mode == "rule1_smallloop"
    assert records[0].parsed_output is not None
    assert records[0].parsed_output.predictions[0].decision_state == "violation"


def test_run_rule1_pipeline_records_errors_and_keeps_going(
    sample_annotation: dict[str, object],
) -> None:
    """One sample failure should not abort the whole Rule 1 run."""
    sample_a = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"image-a", "path": "a.jpg"},
        }
    )
    sample_b = parse_sample(
        {
            **sample_annotation,
            "image_id": "0000425",
            "image": {"bytes": b"image-b", "path": "b.jpg"},
        }
    )
    records = run_rule1_pipeline(
        target_samples=(sample_a, sample_b),
        pipeline=FakeRule1Pipeline(
            {
                sample_a.image_id: RuntimeError("detector failed"),
                sample_b.image_id: _build_image_prediction(
                    image_id=sample_b.image_id,
                    decision_state="unknown",
                ),
            }
        ),
    )

    assert len(records) == 2
    assert records[0].parsed_output is None
    assert records[0].error_message == "detector failed"
    assert records[1].parsed_output is not None
    assert records[1].parsed_output.predictions[0].decision_state == "unknown"


def test_run_rule1_pipeline_writes_progress_and_checkpoint(
    sample_annotation: dict[str, object],
    tmp_path: Path,
) -> None:
    """The runner should emit progress and partial checkpoints during long runs."""
    sample_a = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"image-a", "path": "a.jpg"},
        }
    )
    sample_b = parse_sample(
        {
            **sample_annotation,
            "image_id": "0000425",
            "image": {"bytes": b"image-b", "path": "b.jpg"},
        }
    )
    progress_path = tmp_path / "progress.json"
    checkpoint_path = tmp_path / "checkpoint.json"

    records = run_rule1_pipeline(
        target_samples=(sample_a, sample_b),
        pipeline=FakeRule1Pipeline(
            {
                sample_a.image_id: _build_image_prediction(
                    image_id=sample_a.image_id,
                    decision_state="violation",
                ),
                sample_b.image_id: _build_image_prediction(
                    image_id=sample_b.image_id,
                    decision_state="unknown",
                ),
            }
        ),
        progress_output=progress_path,
        checkpoint_output=checkpoint_path,
        checkpoint_every=1,
    )

    assert len(records) == 2
    progress = read_json(progress_path)
    assert progress["completed"] == 2
    assert progress["total"] == 2
    assert progress["last_image_id"] == sample_b.image_id
    checkpoint = read_json(checkpoint_path)
    assert len(checkpoint) == 2
    assert checkpoint[0]["image_id"] == sample_a.image_id
