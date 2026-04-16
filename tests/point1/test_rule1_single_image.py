"""Tests for single-image Rule 1 helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from common.schemas.bbox import NormalizedBBox
from common.schemas.point1 import Point1Prediction
from point1.pipelines.single_image import (
    SingleImageCandidatePrediction,
    SingleImageSource,
    build_single_image_sample,
    download_image_to_path,
    render_rule1_visualization,
)


def _write_demo_image(path: Path, *, color: tuple[int, int, int] = (255, 255, 255)) -> Path:
    image = Image.new("RGB", (100, 100), color=color)
    image.save(path)
    return path


def test_build_single_image_sample_loads_image_bytes(tmp_path: Path) -> None:
    """Single-image helper should wrap a raw image as a benchmark-style sample."""
    image_path = _write_demo_image(tmp_path / "demo.jpg", color=(12, 34, 56))

    sample = build_single_image_sample(
        image_path,
        image_id="demo-image",
        image_caption="demo caption",
    )

    assert sample.image_id == "demo-image"
    assert sample.image is not None
    assert sample.image.path == str(image_path)
    assert sample.image.bytes == image_path.read_bytes()
    assert sample.image_caption == "demo caption"


def test_download_image_to_path_writes_response_bytes(tmp_path: Path, monkeypatch) -> None:
    """URL downloads should be persisted to a caller-controlled local file."""
    target_path = tmp_path / "downloaded.jpg"
    expected_bytes = b"fake-image-bytes"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def read(self) -> bytes:
            return expected_bytes

    monkeypatch.setattr(
        "point1.pipelines.single_image.urlopen",
        lambda url, timeout=30: FakeResponse(),
    )

    written_path = download_image_to_path("https://example.com/demo.jpg", target_path)

    assert written_path == target_path
    assert target_path.read_bytes() == expected_bytes


def test_render_rule1_visualization_draws_candidate_and_selected_boxes(tmp_path: Path) -> None:
    """Visualization helper should paint candidate boxes onto a saved image."""
    image_path = _write_demo_image(tmp_path / "demo.jpg")
    output_path = tmp_path / "demo.visualized.jpg"
    candidate_prediction = SingleImageCandidatePrediction(
        candidate_id="person-1",
        candidate_bbox=NormalizedBBox(0.1, 0.1, 0.4, 0.4),
        candidate_score=0.9,
        prediction=Point1Prediction(
            rule_id=1,
            decision_state="violation",
            target_bbox=NormalizedBBox(0.5, 0.5, 0.9, 0.9),
            supporting_evidence_ids=("person-1:hard_hat_visible",),
            counter_evidence_ids=(),
            unknown_items=(),
            reason_slots={"subject": "worker candidate 1"},
            reason_text="missing hard hat",
            confidence=0.9,
        ),
    )
    source = SingleImageSource(
        input_mode="path",
        local_path=str(image_path),
        image_id="demo",
        original_path=str(image_path),
    )

    render_rule1_visualization(
        image_path=image_path,
        image_source=source,
        candidate_predictions=(candidate_prediction,),
        image_prediction=candidate_prediction.prediction,
        output_path=output_path,
    )

    assert output_path.exists()
    rendered = Image.open(output_path).convert("RGB")
    assert rendered.getpixel((10, 10)) != (255, 255, 255)
    assert rendered.getpixel((50, 50)) != (255, 255, 255)
