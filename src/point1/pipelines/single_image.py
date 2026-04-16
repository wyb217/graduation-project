"""Helpers for running Rule 1 on one raw image and visualizing the result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

from benchmark.constructionsite10k.types import (
    ConstructionSiteSample,
    SampleAttributes,
    SampleImage,
)
from common.schemas.bbox import NormalizedBBox
from common.schemas.point1 import Point1ImagePredictionSet, Point1Prediction


@dataclass(frozen=True, slots=True)
class SingleImageSource:
    """Describe where a one-off image came from and where it is stored locally."""

    input_mode: Literal["path", "url"]
    local_path: str
    image_id: str
    original_path: str | None = None
    original_url: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "input_mode": self.input_mode,
            "local_path": self.local_path,
            "image_id": self.image_id,
            "original_path": self.original_path,
            "original_url": self.original_url,
        }


@dataclass(frozen=True, slots=True)
class SingleImageCandidatePrediction:
    """A Rule 1 decision plus detector metadata for one candidate box."""

    candidate_id: str
    candidate_bbox: NormalizedBBox
    candidate_score: float
    prediction: Point1Prediction

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "candidate_bbox": self.candidate_bbox.to_list(),
            "candidate_score": self.candidate_score,
            "prediction": self.prediction.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SingleImageRule1Output:
    """Structured payload emitted by the single-image Rule 1 runner."""

    image_source: SingleImageSource
    provider_name: str
    model_name: str
    mode: str
    candidate_backend: str
    predicate_backend: str
    candidate_predictions: tuple[SingleImageCandidatePrediction, ...]
    image_prediction: Point1Prediction
    prediction_set: Point1ImagePredictionSet
    visualization_output: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "image_source": self.image_source.to_dict(),
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "mode": self.mode,
            "candidate_backend": self.candidate_backend,
            "predicate_backend": self.predicate_backend,
            "candidate_count": len(self.candidate_predictions),
            "candidate_predictions": [
                candidate_prediction.to_dict()
                for candidate_prediction in self.candidate_predictions
            ],
            "image_prediction": self.image_prediction.to_dict(),
            "prediction_set": self.prediction_set.to_dict(),
            "visualization_output": self.visualization_output,
        }


def download_image_to_path(url: str, output_path: Path, *, timeout: int = 30) -> Path:
    """Download one image URL to a deterministic local path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response:
        output_path.write_bytes(response.read())
    return output_path


def build_single_image_sample(
    image_path: Path,
    *,
    image_id: str | None = None,
    image_caption: str | None = None,
) -> ConstructionSiteSample:
    """Wrap one raw image file in the benchmark-style sample contract."""
    resolved_path = image_path.resolve()
    effective_image_id = image_id or resolved_path.stem
    effective_caption = image_caption or f"Single-image Rule 1 run for {resolved_path.name}"
    image_bytes = resolved_path.read_bytes()
    return ConstructionSiteSample(
        image_id=effective_image_id,
        image=SampleImage(path=str(resolved_path), bytes=image_bytes),
        image_caption=effective_caption,
        attributes=SampleAttributes(
            illumination="unknown",
            camera_distance="unknown",
            view="unknown",
            quality_of_info="unknown",
        ),
        violations={1: None, 2: None, 3: None, 4: None},
        object_boxes={},
    )


def build_single_image_output(
    *,
    image_source: SingleImageSource,
    provider_name: str,
    model_name: str,
    mode: str,
    candidate_backend: str,
    predicate_backend: str,
    candidate_predictions: tuple[SingleImageCandidatePrediction, ...],
    image_prediction: Point1Prediction,
    prediction_set: Point1ImagePredictionSet,
    visualization_output: str | None,
) -> SingleImageRule1Output:
    """Bundle one-off Rule 1 outputs into a stable JSON contract."""
    return SingleImageRule1Output(
        image_source=image_source,
        provider_name=provider_name,
        model_name=model_name,
        mode=mode,
        candidate_backend=candidate_backend,
        predicate_backend=predicate_backend,
        candidate_predictions=candidate_predictions,
        image_prediction=image_prediction,
        prediction_set=prediction_set,
        visualization_output=visualization_output,
    )


def render_rule1_visualization(
    *,
    image_path: Path,
    image_source: SingleImageSource,
    candidate_predictions: tuple[SingleImageCandidatePrediction, ...],
    image_prediction: Point1Prediction,
    output_path: Path,
) -> Path:
    """Draw candidate boxes plus the selected image-level bbox onto the source image."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    header = (
        f"{image_source.image_id} | overall={image_prediction.decision_state} "
        f"conf={image_prediction.confidence:.3f}"
    )
    _draw_label(draw, font, (8, 8), header, fill=(30, 30, 30), text_fill=(255, 255, 255))

    for candidate_prediction in candidate_predictions:
        color = _decision_color(candidate_prediction.prediction.decision_state)
        pixel_bbox = _normalized_to_pixel_bbox(
            candidate_prediction.candidate_bbox,
            image_width=image.width,
            image_height=image.height,
        )
        draw.rectangle(pixel_bbox, outline=color, width=2)
        label = (
            f"{candidate_prediction.candidate_id} "
            f"{candidate_prediction.prediction.decision_state} "
            f"{candidate_prediction.prediction.confidence:.2f}"
        )
        label_origin = (pixel_bbox[0], max(0, pixel_bbox[1] - 14))
        _draw_label(draw, font, label_origin, label, fill=color, text_fill=(0, 0, 0))

    if image_prediction.target_bbox is not None:
        overall_bbox = _normalized_to_pixel_bbox(
            image_prediction.target_bbox,
            image_width=image.width,
            image_height=image.height,
        )
        highlight_color = (0, 102, 255)
        draw.rectangle(overall_bbox, outline=highlight_color, width=4)
        _draw_label(
            draw,
            font,
            (overall_bbox[0], min(image.height - 14, overall_bbox[3] + 2)),
            "image-level selection",
            fill=highlight_color,
            text_fill=(255, 255, 255),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _normalized_to_pixel_bbox(
    bbox: NormalizedBBox,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left = round(bbox.x_min * image_width)
    top = round(bbox.y_min * image_height)
    right = max(left + 1, round(bbox.x_max * image_width))
    bottom = max(top + 1, round(bbox.y_max * image_height))
    return left, top, right, bottom


def _decision_color(decision_state: str) -> tuple[int, int, int]:
    if decision_state == "violation":
        return (220, 20, 60)
    if decision_state == "no_violation":
        return (34, 139, 34)
    return (255, 191, 0)


def _draw_label(draw, font, origin, text: str, *, fill, text_fill) -> None:  # noqa: ANN001
    x, y = origin
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=fill)
    draw.text((x, y), text, fill=text_fill, font=font)
