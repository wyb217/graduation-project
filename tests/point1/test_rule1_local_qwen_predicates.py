"""Tests for the local-Qwen-backed Rule 1 predicate extractor."""

from __future__ import annotations

import io

from benchmark.constructionsite10k.parser import parse_sample
from common.schemas.bbox import NormalizedBBox
from point1.candidates.person import PersonCandidate
from point1.predicates.rule1_local_qwen import LocalQwenRule1PredicateExtractor


class FakeLocalQwenClient:
    """Fake local Qwen client returning one canned predicate response."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def complete(self, *, messages) -> str:  # noqa: ANN001
        self.calls.append({"messages": messages})
        return self.response_text


def _valid_png_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (128, 128), color=(240, 200, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_local_qwen_rule1_predicate_extractor_parses_json_response(
    sample_annotation: dict[str, object],
) -> None:
    """The local-Qwen extractor should map model JSON into a Rule1PredicateSet."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": _valid_png_bytes(), "path": "demo.png"},
        }
    )
    candidate = PersonCandidate(
        candidate_id="person-1",
        bbox=NormalizedBBox(0.1, 0.2, 0.3, 0.6),
        score=0.9,
    )
    client = FakeLocalQwenClient(
        """
        {
          "person_visible": {"state": "yes", "score": 0.95, "reason": "worker is visible"},
          "ppe_applicable": {"state": "yes", "score": 0.92, "reason": "worker is on foot"},
          "head_region_visible": {"state": "yes", "score": 0.88, "reason": "head is visible"},
          "hard_hat_visible": {"state": "no", "score": 0.81, "reason": "no hard hat"},
          "upper_body_covered": {"state": "yes", "score": 0.77, "reason": "jacket visible"},
          "lower_body_covered": {"state": "unknown", "score": 0.42, "reason": "legs occluded"},
          "toe_covered": {"state": "yes", "score": 0.73, "reason": "shoes visible"}
        }
        """
    )
    extractor = LocalQwenRule1PredicateExtractor(client=client, model_name="demo-model")

    predicates = extractor.extract(sample, candidate)

    assert predicates.candidate_id == "person-1"
    assert predicates.ppe_applicable.state == "yes"
    assert predicates.head_region_visible.state == "yes"
    assert predicates.hard_hat_visible.state == "no"
    assert predicates.lower_body_covered.state == "unknown"
    assert predicates.hard_hat_visible.evidence_bbox == candidate.bbox
    assert len(client.calls) == 1
    content = client.calls[0]["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"
    assert "may also violate Rule 2, 3, or 4" in content[1]["text"]
    assert "Do not set ppe_applicable to no" in content[1]["text"]


def test_local_qwen_rule1_predicate_extractor_short_circuits_tiny_candidates(
    sample_annotation: dict[str, object],
) -> None:
    """Tiny person crops should become unknown without loading the local model."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": _valid_png_bytes(), "path": "demo.png"},
        }
    )
    candidate = PersonCandidate(
        candidate_id="person-1",
        bbox=NormalizedBBox(0.1, 0.2, 0.12, 0.25),
        score=0.9,
    )
    client = FakeLocalQwenClient("{}")
    extractor = LocalQwenRule1PredicateExtractor(
        client=client,
        model_name="demo-model",
        min_candidate_width=24,
        min_candidate_height=48,
    )

    predicates = extractor.extract(sample, candidate)

    assert predicates.person_visible.state == "unknown"
    assert predicates.ppe_applicable.state == "unknown"
    assert predicates.head_region_visible.state == "unknown"
    assert predicates.hard_hat_visible.state == "unknown"
    assert client.calls == []
