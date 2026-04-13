"""Bridge Point 1 outputs into ConstructionSite10k-style prediction payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from common.schemas.bbox import NormalizedBBox
from common.schemas.point1 import (
    Point1BaselineRecord,
    Point1ImagePredictionSet,
    Point1Prediction,
)


@dataclass(frozen=True, slots=True)
class OfficialRuleViolationPrediction:
    """One ConstructionSite10k-style rule prediction."""

    rule_id: int
    bounding_boxes: tuple[NormalizedBBox, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable rule prediction."""
        return {
            "bounding_box": [bbox.to_list() for bbox in self.bounding_boxes],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class OfficialConstructionSitePrediction:
    """A ConstructionSite10k-style prediction record for one image."""

    image_id: str
    rule_predictions: dict[int, OfficialRuleViolationPrediction | None]

    def to_dict(self) -> dict[str, object]:
        """Return the official-style dictionary payload."""
        payload: dict[str, object] = {"image_id": self.image_id}
        for rule_id in range(1, 5):
            rule_prediction = self.rule_predictions.get(rule_id)
            payload[f"rule_{rule_id}_violation"] = (
                None if rule_prediction is None else rule_prediction.to_dict()
            )
        return payload


def build_official_prediction(prediction_set: Point1ImagePredictionSet) -> dict[str, object]:
    """Convert one structured Point 1 output into official-style rule fields."""
    return _build_official_prediction_object(prediction_set).to_dict()


def export_baseline_records_to_official_predictions(
    records: Sequence[Point1BaselineRecord],
) -> list[dict[str, object]]:
    """Convert stored baseline records into official-style prediction dictionaries."""
    return [
        build_official_prediction(record.parsed_output)
        if record.parsed_output is not None
        else build_empty_official_prediction(record.image_id)
        for record in records
    ]


def export_baseline_payload_to_official_predictions(
    payload: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    """Convert baseline JSON payloads into official-style prediction dictionaries."""
    return export_baseline_records_to_official_predictions(
        [_baseline_record_from_dict(record) for record in payload]
    )


def build_empty_official_prediction(image_id: str) -> dict[str, object]:
    """Return an official-style all-null record for a failed parse."""
    return OfficialConstructionSitePrediction(
        image_id=image_id,
        rule_predictions={rule_id: None for rule_id in range(1, 5)},
    ).to_dict()


def _build_official_prediction_object(
    prediction_set: Point1ImagePredictionSet,
) -> OfficialConstructionSitePrediction:
    rule_predictions = {rule_id: None for rule_id in range(1, 5)}
    grouped_predictions: dict[int, list[Point1Prediction]] = {
        rule_id: [] for rule_id in range(1, 5)
    }

    for prediction in prediction_set.predictions:
        if prediction.decision_state != "violation":
            continue
        grouped_predictions[prediction.rule_id].append(prediction)

    for rule_id, predictions in grouped_predictions.items():
        if not predictions:
            continue
        best_prediction = max(predictions, key=lambda prediction: prediction.confidence)
        rule_predictions[rule_id] = OfficialRuleViolationPrediction(
            rule_id=rule_id,
            bounding_boxes=tuple(
                prediction.target_bbox
                for prediction in predictions
                if prediction.target_bbox is not None
            ),
            reason=best_prediction.reason_text,
        )

    return OfficialConstructionSitePrediction(
        image_id=prediction_set.image_id,
        rule_predictions=rule_predictions,
    )


def _baseline_record_from_dict(raw_record: Mapping[str, Any]) -> Point1BaselineRecord:
    parsed_output = raw_record.get("parsed_output")
    return Point1BaselineRecord(
        image_id=str(raw_record["image_id"]),
        provider_name=str(raw_record.get("provider_name", "")),
        model_name=str(raw_record.get("model_name", "")),
        mode=str(raw_record.get("mode", "")),
        raw_response_text=str(raw_record.get("raw_response_text", "")),
        parsed_output=None if parsed_output is None else _prediction_set_from_dict(parsed_output),
        error_message=None
        if raw_record.get("error_message") is None
        else str(raw_record["error_message"]),
    )


def _prediction_set_from_dict(raw_prediction_set: Any) -> Point1ImagePredictionSet:
    if not isinstance(raw_prediction_set, Mapping):
        raise ValueError("parsed_output must be a mapping.")
    raw_predictions = raw_prediction_set.get("predictions", [])
    if not isinstance(raw_predictions, list):
        raise ValueError("parsed_output.predictions must be a list.")
    return Point1ImagePredictionSet(
        image_id=str(raw_prediction_set["image_id"]),
        predictions=tuple(
            _prediction_from_dict(raw_prediction) for raw_prediction in raw_predictions
        ),
    )


def _prediction_from_dict(raw_prediction: Any) -> Point1Prediction:
    if not isinstance(raw_prediction, Mapping):
        raise ValueError("prediction must be a mapping.")
    target_bbox = raw_prediction.get("target_bbox")
    return Point1Prediction(
        rule_id=int(raw_prediction["rule_id"]),
        decision_state=str(raw_prediction["decision_state"]),
        target_bbox=None
        if target_bbox is None
        else NormalizedBBox.from_list([float(value) for value in target_bbox]),
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
