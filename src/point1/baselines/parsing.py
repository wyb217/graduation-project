"""Parsers for raw model responses returned by API baselines."""

from __future__ import annotations

import json
import re
from typing import Any

from benchmark.constructionsite10k.image_info import get_jpeg_dimensions
from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.schemas.bbox import NormalizedBBox
from common.schemas.point1 import Point1ImagePredictionSet, Point1Prediction


def parse_prediction_set_response(
    response_text: str,
    *,
    sample: ConstructionSiteSample | None = None,
) -> Point1ImagePredictionSet:
    """Parse one raw model response into a validated structured prediction set."""
    payload = _load_json_payload(response_text)
    predictions = tuple(_parse_prediction(item, sample=sample) for item in payload["predictions"])
    if len(predictions) != 4:
        raise ValueError(f"Expected 4 predictions, got {len(predictions)}.")
    return Point1ImagePredictionSet(image_id=str(payload["image_id"]), predictions=predictions)


def _load_json_payload(response_text: str) -> dict[str, Any]:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
        if match is None:
            raise ValueError("Could not find JSON object inside markdown fence.")
        stripped = match.group(1)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def _parse_prediction(
    raw_prediction: dict[str, Any],
    *,
    sample: ConstructionSiteSample | None,
) -> Point1Prediction:
    raw_bbox = raw_prediction.get("target_bbox")
    bbox = None if raw_bbox is None else _parse_bbox(raw_bbox, sample=sample)
    return Point1Prediction(
        rule_id=int(raw_prediction["rule_id"]),
        decision_state=str(raw_prediction["decision_state"]),
        target_bbox=bbox,
        supporting_evidence_ids=tuple(
            str(item) for item in raw_prediction["supporting_evidence_ids"]
        ),
        counter_evidence_ids=tuple(str(item) for item in raw_prediction["counter_evidence_ids"]),
        unknown_items=tuple(str(item) for item in raw_prediction["unknown_items"]),
        reason_slots={
            str(key): str(value) for key, value in raw_prediction["reason_slots"].items()
        },
        reason_text=str(raw_prediction["reason_text"]),
        confidence=float(raw_prediction["confidence"]),
    )


def _parse_bbox(raw_bbox: Any, *, sample: ConstructionSiteSample | None) -> NormalizedBBox:
    values = [float(value) for value in raw_bbox]
    try:
        return NormalizedBBox.from_list(values)
    except ValueError:
        normalized = _try_normalize_pixel_bbox(values, sample=sample)
        if normalized is None:
            raise
        return NormalizedBBox.from_list(normalized)


def _try_normalize_pixel_bbox(
    values: list[float],
    *,
    sample: ConstructionSiteSample | None,
) -> list[float] | None:
    if sample is None or sample.image is None or sample.image.bytes is None:
        return None
    dimensions = get_jpeg_dimensions(sample.image.bytes)
    if dimensions is None:
        return None
    width, height = dimensions
    if width <= 0 or height <= 0:
        return None
    return [
        values[0] / width,
        values[1] / height,
        values[2] / width,
        values[3] / height,
    ]
