"""Tests for exporting Point 1 outputs into ConstructionSite10k-style predictions."""

from __future__ import annotations

from common.schemas.bbox import NormalizedBBox
from common.schemas.point1 import Point1BaselineRecord, Point1ImagePredictionSet, Point1Prediction
from eval.bridges.constructionsite10k import (
    build_official_prediction,
    export_baseline_records_to_official_predictions,
)


def test_build_official_prediction_maps_violation_fields() -> None:
    """A structured Point 1 output should map into the four official rule fields."""
    prediction_set = Point1ImagePredictionSet(
        image_id="0000424",
        predictions=(
            Point1Prediction(
                rule_id=1,
                decision_state="violation",
                target_bbox=NormalizedBBox(0.1, 0.2, 0.3, 0.4),
                supporting_evidence_ids=(),
                counter_evidence_ids=(),
                unknown_items=(),
                reason_slots={"subject": "worker"},
                reason_text="Worker without hard hat.",
                confidence=0.9,
            ),
            Point1Prediction(
                rule_id=2,
                decision_state="no_violation",
                target_bbox=None,
                supporting_evidence_ids=(),
                counter_evidence_ids=(),
                unknown_items=(),
                reason_slots={},
                reason_text="",
                confidence=0.8,
            ),
            Point1Prediction(
                rule_id=3,
                decision_state="no_violation",
                target_bbox=None,
                supporting_evidence_ids=(),
                counter_evidence_ids=(),
                unknown_items=(),
                reason_slots={},
                reason_text="",
                confidence=0.8,
            ),
            Point1Prediction(
                rule_id=4,
                decision_state="no_violation",
                target_bbox=None,
                supporting_evidence_ids=(),
                counter_evidence_ids=(),
                unknown_items=(),
                reason_slots={},
                reason_text="",
                confidence=0.8,
            ),
        ),
    )

    official_prediction = build_official_prediction(prediction_set)

    assert official_prediction["image_id"] == "0000424"
    assert official_prediction["rule_1_violation"] == {
        "bounding_box": [[0.1, 0.2, 0.3, 0.4]],
        "reason": "Worker without hard hat.",
    }
    assert official_prediction["rule_2_violation"] is None
    assert official_prediction["rule_3_violation"] is None
    assert official_prediction["rule_4_violation"] is None


def test_export_baseline_records_uses_null_predictions_for_parse_failures() -> None:
    """A failed baseline parse should still export a complete all-null official record."""
    records = [
        Point1BaselineRecord(
            image_id="broken-sample",
            provider_name="demo",
            model_name="demo-model",
            mode="direct",
            raw_response_text="not json",
            parsed_output=None,
            error_message="bad response",
        )
    ]

    official_predictions = export_baseline_records_to_official_predictions(records)

    assert official_predictions == [
        {
            "image_id": "broken-sample",
            "rule_1_violation": None,
            "rule_2_violation": None,
            "rule_3_violation": None,
            "rule_4_violation": None,
        }
    ]
