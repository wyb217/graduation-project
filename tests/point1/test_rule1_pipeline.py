"""Tests for the Rule 1 candidate -> predicate -> executor pipeline."""

from __future__ import annotations

from benchmark.constructionsite10k.parser import parse_sample
from common.schemas.bbox import NormalizedBBox
from point1.candidates.person import PersonCandidate
from point1.pipelines.rule1 import Rule1Pipeline
from point1.predicates.rule1 import Rule1PredicateResult, Rule1PredicateSet


class FakeCandidateGenerator:
    """Return pre-defined person candidates for pipeline tests."""

    def __init__(self, candidates: tuple[PersonCandidate, ...]) -> None:
        self.candidates = candidates

    def generate(self, sample) -> tuple[PersonCandidate, ...]:  # noqa: ANN001
        return self.candidates


class FakePredicateExtractor:
    """Return pre-defined predicate sets keyed by candidate id."""

    def __init__(self, predicate_sets: dict[str, Rule1PredicateSet]) -> None:
        self.predicate_sets = predicate_sets

    def extract(self, sample, candidate: PersonCandidate) -> Rule1PredicateSet:  # noqa: ANN001
        return self.predicate_sets[candidate.candidate_id]


def _predicate(state: str, score: float, reason: str) -> Rule1PredicateResult:
    return Rule1PredicateResult(
        state=state,
        score=score,
        reason=reason,
        evidence_bbox=None,
    )


def test_rule1_pipeline_runs_candidates_through_executor(
    sample_annotation: dict[str, object],
) -> None:
    """The pipeline should return one structured Rule 1 prediction per person candidate."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"fake-image", "path": "demo.jpg"},
        }
    )
    candidates = (
        PersonCandidate("person-1", NormalizedBBox(0.1, 0.2, 0.3, 0.6), 0.9),
        PersonCandidate("person-2", NormalizedBBox(0.4, 0.2, 0.6, 0.7), 0.85),
    )
    predicate_sets = {
        "person-1": Rule1PredicateSet(
            candidate_id="person-1",
            person_visible=_predicate("yes", 0.95, "person is clearly visible"),
            hard_hat_visible=_predicate("no", 0.9, "no hard hat is visible"),
            upper_body_covered=_predicate("yes", 0.8, "upper body is covered"),
            lower_body_covered=_predicate("yes", 0.8, "lower body is covered"),
            toe_covered=_predicate("yes", 0.8, "toe protection is visible"),
        ),
        "person-2": Rule1PredicateSet(
            candidate_id="person-2",
            person_visible=_predicate("yes", 0.95, "person is clearly visible"),
            hard_hat_visible=_predicate("yes", 0.9, "hard hat is visible"),
            upper_body_covered=_predicate("yes", 0.8, "upper body is covered"),
            lower_body_covered=_predicate("yes", 0.8, "lower body is covered"),
            toe_covered=_predicate("yes", 0.8, "toe protection is visible"),
        ),
    }
    pipeline = Rule1Pipeline(
        candidate_generator=FakeCandidateGenerator(candidates),
        predicate_extractor=FakePredicateExtractor(predicate_sets),
    )

    result = pipeline.run(sample)

    assert result.image_id == sample.image_id
    assert len(result.predictions) == 2
    assert result.predictions[0].decision_state == "violation"
    assert result.predictions[1].decision_state == "no_violation"


def test_rule1_pipeline_returns_unknown_when_no_person_candidates_are_found(
    sample_annotation: dict[str, object],
) -> None:
    """The pipeline should emit one image-level unknown when detection finds nobody."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"fake-image", "path": "demo.jpg"},
        }
    )
    pipeline = Rule1Pipeline(
        candidate_generator=FakeCandidateGenerator(()),
        predicate_extractor=FakePredicateExtractor({}),
    )

    result = pipeline.run(sample)

    assert len(result.predictions) == 1
    prediction = result.predictions[0]
    assert prediction.decision_state == "unknown"
    assert prediction.unknown_items == ("person_detection",)
    assert "No reliable person candidate" in prediction.reason_text
