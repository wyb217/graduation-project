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


def test_parse_prediction_set_response_handles_author_sparse_rule_payload() -> None:
    """Official-style sparse rule dictionaries should be adapted back to four predictions."""
    response = """
    {
      "image_id": "0000424",
      "1": {
        "reason": "Worker on the left is not wearing a hard hat.",
        "bounding_box": [0.1, 0.2, 0.3, 0.4]
      },
      "4": {
        "reason": "Worker is standing in the excavator blind spot.",
        "bounding_box": [0.5, 0.6, 0.7, 0.8]
      }
    }
    """

    prediction_set = parse_prediction_set_response(response)

    assert len(prediction_set.predictions) == 4
    assert prediction_set.predictions[0].decision_state == "violation"
    assert (
        prediction_set.predictions[0].reason_text == "Worker on the left is not wearing a hard hat."
    )
    assert prediction_set.predictions[3].decision_state == "violation"
    assert prediction_set.predictions[3].target_bbox is not None


def test_parse_prediction_set_response_handles_author_sparse_no_violation_payload() -> None:
    """Official-style no-violation responses should map to four negative predictions."""
    response = """
    {
      "image_id": "0000424",
      "0": "No violations"
    }
    """

    prediction_set = parse_prediction_set_response(response)

    assert len(prediction_set.predictions) == 4
    assert all(
        prediction.decision_state == "no_violation" for prediction in prediction_set.predictions
    )


def test_parse_prediction_set_response_uses_sample_image_id_for_sparse_author_payload(
    sample_annotation: dict[str, object],
) -> None:
    """Sparse author payloads should fall back to the current sample image_id."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image_id": "target-1",
            "image": {"bytes": b"target-image", "path": "target.jpg"},
        }
    )
    response = """
    {
      "0": "No violations"
    }
    """

    prediction_set = parse_prediction_set_response(response, sample=sample)

    assert prediction_set.image_id == "target-1"
    assert all(
        prediction.decision_state == "no_violation" for prediction in prediction_set.predictions
    )


def test_parse_prediction_set_response_treats_negative_sparse_reasons_as_no_violation() -> None:
    """Sparse author responses with explicit negative reasons should stay negative."""
    response = """
    {
      "image_id": "0000002",
      "1": {
        "reason": "Workers are wearing hard hats and no violations of PPE are visible.",
        "bounding_box": null
      },
      "2": {
        "reason": "No workers are visible working at height, so the rule does not apply.",
        "bounding_box": null
      },
      "3": {
        "reason": "No underground project edge is visible here.",
        "bounding_box": null
      },
      "4": {
        "reason": "No workers are in the excavator operating radius.",
        "bounding_box": null
      }
    }
    """

    prediction_set = parse_prediction_set_response(response)

    assert all(
        prediction.decision_state == "no_violation" for prediction in prediction_set.predictions
    )
