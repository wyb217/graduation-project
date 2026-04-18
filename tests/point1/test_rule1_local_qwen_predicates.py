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


class FakeBatchLocalQwenClient:
    """Fake local Qwen client returning one canned response per candidate."""

    def __init__(self, response_texts: list[str]) -> None:
        self.response_texts = response_texts
        self.batch_calls: list[dict[str, object]] = []

    def complete_batch(self, *, messages_batch) -> list[str]:  # noqa: ANN001
        self.batch_calls.append({"messages_batch": messages_batch})
        return self.response_texts


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
    assert "Candidate bbox (normalized xyxy)" not in content[1]["text"]


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


def test_local_qwen_rule1_predicate_extractor_extract_many_batches_candidates(
    sample_annotation: dict[str, object],
) -> None:
    """The extractor should batch valid candidates and preserve input ordering."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": _valid_png_bytes(), "path": "demo.png"},
        }
    )
    candidates = (
        PersonCandidate(
            candidate_id="person-1",
            bbox=NormalizedBBox(0.1, 0.1, 0.4, 0.8),
            score=0.9,
        ),
        PersonCandidate(
            candidate_id="person-2",
            bbox=NormalizedBBox(0.5, 0.1, 0.8, 0.8),
            score=0.88,
        ),
    )
    client = FakeBatchLocalQwenClient(
        [
            """
            {
              "person_visible": {"state": "yes", "score": 0.95, "reason": "worker 1 visible"},
              "ppe_applicable": {"state": "yes", "score": 0.92, "reason": "worker 1 on foot"},
              "head_region_visible": {"state": "yes", "score": 0.88, "reason": "head 1 visible"},
              "hard_hat_visible": {"state": "no", "score": 0.81, "reason": "no hard hat 1"},
              "upper_body_covered": {"state": "yes", "score": 0.77, "reason": "jacket 1"},
              "lower_body_covered": {
                "state": "unknown",
                "score": 0.42,
                "reason": "legs 1 occluded"
              },
              "toe_covered": {"state": "yes", "score": 0.73, "reason": "shoes 1 visible"}
            }
            """,
            """
            {
              "person_visible": {"state": "yes", "score": 0.90, "reason": "worker 2 visible"},
              "ppe_applicable": {"state": "yes", "score": 0.85, "reason": "worker 2 on foot"},
              "head_region_visible": {"state": "yes", "score": 0.80, "reason": "head 2 visible"},
              "hard_hat_visible": {"state": "yes", "score": 0.75, "reason": "hard hat 2"},
              "upper_body_covered": {"state": "yes", "score": 0.70, "reason": "jacket 2"},
              "lower_body_covered": {"state": "yes", "score": 0.65, "reason": "pants 2 visible"},
              "toe_covered": {"state": "unknown", "score": 0.30, "reason": "feet 2 unclear"}
            }
            """,
        ]
    )
    extractor = LocalQwenRule1PredicateExtractor(
        client=client,
        model_name="demo-model",
        candidate_batch_size=4,
    )

    predicate_sets = extractor.extract_many(sample, candidates)

    assert len(predicate_sets) == 2
    assert predicate_sets[0].candidate_id == "person-1"
    assert predicate_sets[0].hard_hat_visible.state == "no"
    assert predicate_sets[1].candidate_id == "person-2"
    assert predicate_sets[1].toe_covered.state == "unknown"
    assert len(client.batch_calls) == 1
    messages_batch = client.batch_calls[0]["messages_batch"]
    assert isinstance(messages_batch, list)
    assert len(messages_batch) == 2


def test_local_qwen_rule1_predicate_extractor_supports_full_image_context(
    sample_annotation: dict[str, object],
) -> None:
    """The context mode should attach both the crop and the full image to the prompt."""
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
    extractor = LocalQwenRule1PredicateExtractor(
        client=client,
        model_name="demo-model",
        context_mode="crop_with_full_image",
    )

    extractor.extract(sample, candidate)

    content = client.calls[0]["messages"][1]["content"]
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "image"
    assert content[2]["type"] == "text"
    assert "first image is the worker crop" in content[2]["text"]
    assert "Candidate bbox (normalized xyxy)" in content[2]["text"]


def test_local_qwen_rule1_predicate_extractor_can_expand_crop_with_padding_profile(
    sample_annotation: dict[str, object],
) -> None:
    """Padding profiles should only change crop geometry, not crop-only prompt semantics."""
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
    response = """
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
    client_none = FakeLocalQwenClient(response)
    extractor_none = LocalQwenRule1PredicateExtractor(
        client=client_none,
        model_name="demo-model",
        crop_padding_profile="none",
    )
    client_padded = FakeLocalQwenClient(response)
    extractor_padded = LocalQwenRule1PredicateExtractor(
        client=client_padded,
        model_name="demo-model",
        crop_padding_profile="rule1_ppe",
    )

    extractor_none.extract(sample, candidate)
    extractor_padded.extract(sample, candidate)

    none_content = client_none.calls[0]["messages"][1]["content"]
    padded_content = client_padded.calls[0]["messages"][1]["content"]
    assert none_content[0]["image"].size == (26, 51)
    assert padded_content[0]["image"].size == (30, 68)
    assert "Candidate bbox (normalized xyxy)" not in padded_content[1]["text"]


def test_local_qwen_rule1_predicate_extractor_resizes_large_crops_before_inference(
    sample_annotation: dict[str, object],
) -> None:
    """Large candidate crops should be downscaled to the configured longest-side cap."""
    from PIL import Image

    image = Image.new("RGB", (2000, 1000), color=(240, 200, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": buffer.getvalue(), "path": "demo-large.png"},
        }
    )
    candidate = PersonCandidate(
        candidate_id="person-1",
        bbox=NormalizedBBox(0.1, 0.1, 0.9, 0.9),
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
    extractor = LocalQwenRule1PredicateExtractor(
        client=client,
        model_name="demo-model",
        max_crop_longest_side=640,
    )

    extractor.extract(sample, candidate)

    content = client.calls[0]["messages"][1]["content"]
    resized_crop = content[0]["image"]
    assert max(resized_crop.size) == 640
    assert resized_crop.size == (640, 320)


def test_local_qwen_rule1_predicate_extractor_does_not_upscale_small_crops(
    sample_annotation: dict[str, object],
) -> None:
    """Small candidate crops should keep their original size when below the cap."""
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
    extractor = LocalQwenRule1PredicateExtractor(
        client=client,
        model_name="demo-model",
        max_crop_longest_side=640,
    )

    extractor.extract(sample, candidate)

    content = client.calls[0]["messages"][1]["content"]
    assert content[0]["image"].size == (26, 51)
