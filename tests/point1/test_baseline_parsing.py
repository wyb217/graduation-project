"""Tests for parsing Point 1 baseline model outputs."""

from __future__ import annotations

from benchmark.constructionsite10k.parser import parse_sample
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


def test_parse_prediction_set_response_converts_pixel_bbox_to_normalized(
    sample_annotation: dict[str, object],
) -> None:
    """Pixel-space bbox outputs should be normalized when image dimensions are available."""
    jpeg_bytes = (
        b"\xff\xd8\xff\xc0\x00\x11\x08\x07\x80\x0a\x00\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    )
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {
                "bytes": jpeg_bytes,
                "path": "demo.jpg",
            },
        }
    )
    response = """
    {
      "image_id": "0000424",
      "predictions": [
        {
          "rule_id": 1,
          "decision_state": "violation",
          "target_bbox": [100, 200, 500, 1000],
          "supporting_evidence_ids": [],
          "counter_evidence_ids": [],
          "unknown_items": [],
          "reason_slots": {},
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
    """

    prediction_set = parse_prediction_set_response(response, sample=sample)

    assert prediction_set.predictions[0].target_bbox is not None
    assert prediction_set.predictions[0].target_bbox.to_list() == [
        0.0390625,
        0.10416666666666667,
        0.1953125,
        0.5208333333333334,
    ]


def test_parse_prediction_set_response_handles_author_style_payload() -> None:
    """Author-style five-shot outputs should be adapted back to four rule predictions."""
    response = """
    {
      "image_id": "0000424",
      "violated_rule_ids": [1, 3],
      "explanation": "Worker missing hard hat near an unprotected edge.",
      "target_bbox": [0.1, 0.2, 0.3, 0.4]
    }
    """

    prediction_set = parse_prediction_set_response(response)

    assert len(prediction_set.predictions) == 4
    assert prediction_set.predictions[0].decision_state == "violation"
    assert prediction_set.predictions[1].decision_state == "no_violation"
    assert prediction_set.predictions[2].decision_state == "violation"
    assert prediction_set.predictions[0].target_bbox is not None
