"""Tests for parsing Point 1 baseline model outputs."""

from __future__ import annotations

from point1.baselines.parsing import parse_prediction_set_response


def test_parse_prediction_set_response_handles_fenced_json() -> None:
    """The parser should extract JSON even when a provider wraps it in a markdown fence."""
    response = """```json
{
  "image_id": "0000424",
  "predictions": [
    {
      "rule_id": 1,
      "decision_state": "violation",
      "target_bbox": [0.1, 0.2, 0.3, 0.4],
      "supporting_evidence_ids": [],
      "counter_evidence_ids": [],
      "unknown_items": [],
      "reason_slots": {"subject": "worker"},
      "reason_text": "worker missing hard hat",
      "confidence": 0.8
    },
    {
      "rule_id": 2,
      "decision_state": "no_violation",
      "target_bbox": null,
      "supporting_evidence_ids": [],
      "counter_evidence_ids": [],
      "unknown_items": [],
      "reason_slots": {},
      "reason_text": "no visible violation",
      "confidence": 0.7
    },
    {
      "rule_id": 3,
      "decision_state": "unknown",
      "target_bbox": null,
      "supporting_evidence_ids": [],
      "counter_evidence_ids": [],
      "unknown_items": ["visibility"],
      "reason_slots": {},
      "reason_text": "unclear",
      "confidence": 0.2
    },
    {
      "rule_id": 4,
      "decision_state": "no_violation",
      "target_bbox": null,
      "supporting_evidence_ids": [],
      "counter_evidence_ids": [],
      "unknown_items": [],
      "reason_slots": {},
      "reason_text": "no excavator risk",
      "confidence": 0.9
    }
  ]
}
```"""

    prediction_set = parse_prediction_set_response(response)

    assert prediction_set.image_id == "0000424"
    assert len(prediction_set.predictions) == 4
    assert prediction_set.predictions[0].target_bbox is not None
    assert prediction_set.predictions[2].decision_state == "unknown"
