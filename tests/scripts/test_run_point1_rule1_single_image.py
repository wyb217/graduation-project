"""CLI tests for the single-image Rule 1 runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image

from common.schemas.bbox import NormalizedBBox
from common.schemas.point1 import Point1Prediction
from point1.candidates.person import PersonCandidate
from point1.pipelines.rule1 import Rule1PipelineResult


def load_single_image_script_module():
    """Load the single-image runner script as a module."""
    module_path = Path("scripts/run_point1_rule1_single_image.py")
    spec = importlib.util.spec_from_file_location(
        "run_point1_rule1_single_image_script", module_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load scripts/run_point1_rule1_single_image.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_demo_image(path: Path) -> Path:
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(path)
    return path


def test_run_point1_rule1_single_image_writes_json_and_visualization(tmp_path: Path) -> None:
    """The single-image runner should emit both JSON and a bbox visualization."""
    module = load_single_image_script_module()
    image_path = _write_demo_image(tmp_path / "demo.jpg")
    output_path = tmp_path / "demo.rule1.json"
    visualization_path = tmp_path / "demo.visualized.jpg"

    candidate = PersonCandidate(
        candidate_id="person-1",
        bbox=NormalizedBBox(0.1, 0.2, 0.5, 0.9),
        score=0.95,
    )
    prediction = Point1Prediction(
        rule_id=1,
        decision_state="violation",
        target_bbox=NormalizedBBox(0.1, 0.2, 0.5, 0.9),
        supporting_evidence_ids=("person-1:hard_hat_visible",),
        counter_evidence_ids=(),
        unknown_items=(),
        reason_slots={"subject": "worker candidate 1", "missing_item": "hard_hat"},
        reason_text="worker candidate 1 is on foot at the site without hard hat.",
        confidence=0.95,
    )

    class FakePipeline:
        def run(self, sample):  # noqa: ANN001
            return Rule1PipelineResult(
                image_id=sample.image_id,
                candidate_predictions=(prediction,),
                image_prediction=prediction,
            )

    class FakeCandidateGenerator:
        def generate(self, sample):  # noqa: ANN001
            return (candidate,)

    module._build_rule1_runtime = lambda args: (
        FakePipeline(),
        "rule1_pipeline_local_qwen",
        "/models/qwen3-vl",
        "rule1_single_image_local_qwen",
    )
    module._build_candidate_generator = lambda args: FakeCandidateGenerator()

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_rule1_single_image.py",
            "--image-path",
            str(image_path),
            "--output",
            str(output_path),
            "--visualization-output",
            str(visualization_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    import json

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 1
    assert payload["candidate_predictions"][0]["candidate_bbox"] == [0.1, 0.2, 0.5, 0.9]
    assert payload["image_prediction"]["decision_state"] == "violation"
    assert payload["visualization_output"] == str(visualization_path)
    assert visualization_path.exists()
