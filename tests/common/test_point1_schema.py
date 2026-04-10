"""Tests for structured Point 1 outputs."""

from __future__ import annotations

from common.schemas.point1 import Point1Prediction


def test_point1_prediction_exposes_structured_payload() -> None:
    """Point 1 outputs should be explicit and machine-readable."""
    prediction = Point1Prediction(
        rule_id=1,
        decision_state="violation",
        target_bbox=None,
        supporting_evidence_ids=("e1",),
        counter_evidence_ids=(),
        unknown_items=("head_visibility",),
        reason_slots={"subject": "worker"},
        reason_text="worker missing hard hat",
        confidence=0.8,
    )

    assert prediction.reason_slots["subject"] == "worker"
    assert prediction.supporting_evidence_ids == ("e1",)
