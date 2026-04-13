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
    if "predictions" in payload:
        predictions = tuple(
            _parse_prediction(item, sample=sample) for item in payload["predictions"]
        )
    elif "violated_rule_ids" in payload:
        predictions = _parse_author_style_payload(payload, sample=sample)
    else:
        predictions = _parse_author_sparse_rule_payload(payload, sample=sample)
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


def _parse_author_style_payload(
    payload: dict[str, Any],
    *,
    sample: ConstructionSiteSample | None,
) -> tuple[Point1Prediction, ...]:
    violated_rule_ids = {int(rule_id) for rule_id in payload.get("violated_rule_ids", [])}
    explanation = str(payload.get("explanation", "")).strip()
    raw_bbox = payload.get("target_bbox")
    bbox = None if raw_bbox is None else _parse_bbox(raw_bbox, sample=sample)
    predictions: list[Point1Prediction] = []
    for rule_id in range(1, 5):
        is_violated = rule_id in violated_rule_ids
        predictions.append(
            Point1Prediction(
                rule_id=rule_id,
                decision_state="violation" if is_violated else "no_violation",
                target_bbox=bbox if is_violated else None,
                supporting_evidence_ids=(),
                counter_evidence_ids=(),
                unknown_items=(),
                reason_slots={},
                reason_text=explanation if is_violated else "",
                confidence=float(payload.get("confidence", 1.0 if violated_rule_ids else 0.0)),
            )
        )
    return tuple(predictions)


def _parse_author_sparse_rule_payload(
    payload: dict[str, Any],
    *,
    sample: ConstructionSiteSample | None,
) -> tuple[Point1Prediction, ...]:
    sparse_predictions = payload.copy()
    sparse_predictions.pop("image_id", None)

    if "0" in sparse_predictions:
        sparse_predictions = {}

    predictions: list[Point1Prediction] = []
    for rule_id in range(1, 5):
        raw_prediction = sparse_predictions.get(str(rule_id))
        if raw_prediction is None:
            predictions.append(
                Point1Prediction(
                    rule_id=rule_id,
                    decision_state="no_violation",
                    target_bbox=None,
                    supporting_evidence_ids=(),
                    counter_evidence_ids=(),
                    unknown_items=(),
                    reason_slots={},
                    reason_text="",
                    confidence=0.0,
                )
            )
            continue

        reason_text = ""
        raw_bbox: Any = None
        if isinstance(raw_prediction, dict):
            reason_text = str(raw_prediction.get("reason", "")).strip()
            raw_bbox = _extract_sparse_bbox(raw_prediction)
        else:
            reason_text = str(raw_prediction).strip()
        bbox = None if raw_bbox is None else _parse_bbox(raw_bbox, sample=sample)
        predictions.append(
            Point1Prediction(
                rule_id=rule_id,
                decision_state="violation",
                target_bbox=bbox,
                supporting_evidence_ids=(),
                counter_evidence_ids=(),
                unknown_items=(),
                reason_slots={},
                reason_text=reason_text,
                confidence=1.0,
            )
        )
    return tuple(predictions)


def _extract_sparse_bbox(raw_prediction: dict[str, Any]) -> Any:
    for key in ("bounding_box", "bounding box", "bbox"):
        if key in raw_prediction:
            return raw_prediction[key]
    return None
