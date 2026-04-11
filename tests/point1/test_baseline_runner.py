"""Tests for the Point 1 API baseline runner."""

from __future__ import annotations

from benchmark.constructionsite10k.parser import parse_sample
from point1.baselines.runner import run_api_baseline


class FakeVisionClient:
    """Simple fake client used to test the runner without real API calls."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def complete(self, *, messages, model, temperature, max_tokens) -> str:  # noqa: ANN001
        self.calls.append(messages)
        return """
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
              "reason_slots": {},
              "reason_text": "r1",
              "confidence": 0.9
            },
            {
              "rule_id": 2,
              "decision_state": "no_violation",
              "target_bbox": null,
              "supporting_evidence_ids": [],
              "counter_evidence_ids": [],
              "unknown_items": [],
              "reason_slots": {},
              "reason_text": "r2",
              "confidence": 0.9
            },
            {
              "rule_id": 3,
              "decision_state": "no_violation",
              "target_bbox": null,
              "supporting_evidence_ids": [],
              "counter_evidence_ids": [],
              "unknown_items": [],
              "reason_slots": {},
              "reason_text": "r3",
              "confidence": 0.9
            },
            {
              "rule_id": 4,
              "decision_state": "no_violation",
              "target_bbox": null,
              "supporting_evidence_ids": [],
              "counter_evidence_ids": [],
              "unknown_items": [],
              "reason_slots": {},
              "reason_text": "r4",
              "confidence": 0.9
            }
          ]
        }
        """


class FlakyVisionClient:
    """Fake client that fails once and then succeeds."""

    def __init__(self) -> None:
        self.num_calls = 0

    def complete(self, *, messages, model, temperature, max_tokens) -> str:  # noqa: ANN001
        self.num_calls += 1
        if self.num_calls == 1:
            raise RuntimeError("temporary provider failure")
        return """
        {
          "image_id": "0000424",
          "predictions": [
            {
              "rule_id": 1,
              "decision_state": "no_violation",
              "target_bbox": null,
              "supporting_evidence_ids": [],
              "counter_evidence_ids": [],
              "unknown_items": [],
              "reason_slots": {},
              "reason_text": "r1",
              "confidence": 0.9
            },
            {
              "rule_id": 2,
              "decision_state": "no_violation",
              "target_bbox": null,
              "supporting_evidence_ids": [],
              "counter_evidence_ids": [],
              "unknown_items": [],
              "reason_slots": {},
              "reason_text": "r2",
              "confidence": 0.9
            },
            {
              "rule_id": 3,
              "decision_state": "no_violation",
              "target_bbox": null,
              "supporting_evidence_ids": [],
              "counter_evidence_ids": [],
              "unknown_items": [],
              "reason_slots": {},
              "reason_text": "r3",
              "confidence": 0.9
            },
            {
              "rule_id": 4,
              "decision_state": "no_violation",
              "target_bbox": null,
              "supporting_evidence_ids": [],
              "counter_evidence_ids": [],
              "unknown_items": [],
              "reason_slots": {},
              "reason_text": "r4",
              "confidence": 0.9
            }
          ]
        }
        """


def test_run_api_baseline_returns_structured_results(
    sample_annotation: dict[str, object],
) -> None:
    """The runner should return parsed records and preserve raw responses."""
    target_sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"target-image", "path": "target.jpg"},
        }
    )
    client = FakeVisionClient()

    results = run_api_baseline(
        client=client,
        model_name="demo-model",
        provider_name="demo-provider",
        target_samples=(target_sample,),
        mode="direct",
        example_samples=(),
    )

    assert len(results) == 1
    assert results[0].provider_name == "demo-provider"
    assert results[0].parsed_output is not None
    assert results[0].parsed_output.predictions[0].decision_state == "violation"
    assert len(client.calls) == 1


def test_run_api_baseline_keeps_going_after_one_request_failure(
    sample_annotation: dict[str, object],
) -> None:
    """A single provider failure should not abort the whole run."""
    sample_a = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"image-a", "path": "a.jpg"},
        }
    )
    sample_b = parse_sample(
        {
            **sample_annotation,
            "image_id": "0000425",
            "image": {"bytes": b"image-b", "path": "b.jpg"},
        }
    )
    client = FlakyVisionClient()

    results = run_api_baseline(
        client=client,
        model_name="demo-model",
        provider_name="demo-provider",
        target_samples=(sample_a, sample_b),
        mode="direct",
        example_samples=(),
    )

    assert len(results) == 2
    assert results[0].parsed_output is None
    assert results[0].error_message == "temporary provider failure"
    assert results[1].parsed_output is not None
