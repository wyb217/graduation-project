"""Rule 1 pipeline: person candidates -> predicates -> executor -> image-level decision."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.schemas.point1 import Point1ImagePredictionSet, Point1Prediction
from point1.candidates.person import OpenCVHogPersonCandidateGenerator
from point1.executor.rule1 import execute_rule1_candidate
from point1.predicates.rule1 import HeuristicRule1PredicateExtractor


@dataclass(frozen=True, slots=True)
class Rule1PipelineResult:
    """Image-level Rule 1 pipeline output."""

    image_id: str
    candidate_predictions: tuple[Point1Prediction, ...]
    image_prediction: Point1Prediction
    candidate_ms: float = 0.0
    predicate_ms: float = 0.0
    executor_ms: float = 0.0
    total_ms: float = 0.0
    candidate_count: int = 0
    candidate_count_raw: int = 0
    candidate_count_capped: int = 0
    fallback_used: bool = False
    max_candidates_per_image: int | None = None

    def to_prediction_set(self) -> Point1ImagePredictionSet:
        """Return a baseline-compatible image-level prediction payload."""
        return Point1ImagePredictionSet(
            image_id=self.image_id,
            predictions=(self.image_prediction,),
        )


class Rule1Pipeline:
    """Run Rule 1 over one image with a detector and predicate extractor."""

    def __init__(
        self,
        *,
        candidate_generator: OpenCVHogPersonCandidateGenerator | object | None = None,
        predicate_extractor: HeuristicRule1PredicateExtractor | object | None = None,
        max_candidates_per_image: int | None = None,
    ) -> None:
        self._candidate_generator = (
            OpenCVHogPersonCandidateGenerator()
            if candidate_generator is None
            else candidate_generator
        )
        self._predicate_extractor = (
            HeuristicRule1PredicateExtractor()
            if predicate_extractor is None
            else predicate_extractor
        )
        if max_candidates_per_image is not None and max_candidates_per_image <= 0:
            raise ValueError("max_candidates_per_image must be positive when provided.")
        self._max_candidates_per_image = max_candidates_per_image

    def run(self, sample: ConstructionSiteSample) -> Rule1PipelineResult:
        """Return Rule 1 predictions for one benchmark sample."""
        total_start = perf_counter()
        candidate_start = perf_counter()
        candidates = self._candidate_generator.generate(sample)
        candidate_ms = (perf_counter() - candidate_start) * 1000.0
        candidate_count_raw = len(candidates)
        candidates = _cap_candidates(
            candidates,
            max_candidates_per_image=self._max_candidates_per_image,
        )
        candidate_count_capped = len(candidates)
        candidate_count = candidate_count_capped
        fallback_used = bool(getattr(self._candidate_generator, "last_used_fallback", False))
        if not candidates:
            image_prediction = self._build_detection_unknown_prediction()
            return Rule1PipelineResult(
                image_id=sample.image_id,
                candidate_predictions=(),
                image_prediction=image_prediction,
                candidate_ms=candidate_ms,
                predicate_ms=0.0,
                executor_ms=0.0,
                total_ms=(perf_counter() - total_start) * 1000.0,
                candidate_count=candidate_count,
                candidate_count_raw=candidate_count_raw,
                candidate_count_capped=candidate_count_capped,
                fallback_used=fallback_used,
                max_candidates_per_image=self._max_candidates_per_image,
            )

        predicate_start = perf_counter()
        batched_extract = getattr(self._predicate_extractor, "extract_many", None)
        if callable(batched_extract):
            predicate_sets = batched_extract(sample, candidates)
        else:
            predicate_sets = tuple(
                self._predicate_extractor.extract(sample, candidate) for candidate in candidates
            )
        predicate_ms = (perf_counter() - predicate_start) * 1000.0
        executor_start = perf_counter()
        candidate_predictions = tuple(
            execute_rule1_candidate(candidate, predicate_set)
            for candidate, predicate_set in zip(candidates, predicate_sets, strict=True)
        )
        image_prediction = self._aggregate_image_prediction(candidate_predictions)
        executor_ms = (perf_counter() - executor_start) * 1000.0
        return Rule1PipelineResult(
            image_id=sample.image_id,
            candidate_predictions=candidate_predictions,
            image_prediction=image_prediction,
            candidate_ms=candidate_ms,
            predicate_ms=predicate_ms,
            executor_ms=executor_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
            candidate_count=candidate_count,
            candidate_count_raw=candidate_count_raw,
            candidate_count_capped=candidate_count_capped,
            fallback_used=fallback_used,
            max_candidates_per_image=self._max_candidates_per_image,
        )

    def _build_detection_unknown_prediction(self) -> Point1Prediction:
        return Point1Prediction(
            rule_id=1,
            decision_state="unknown",
            target_bbox=None,
            supporting_evidence_ids=(),
            counter_evidence_ids=(),
            unknown_items=("person_detection",),
            reason_slots={
                "subject": "worker candidate",
                "missing_item": "person_detection",
                "scene_condition": "on foot at the construction site",
            },
            reason_text="No reliable person candidate was detected for Rule 1 inspection.",
            confidence=0.0,
        )

    def _aggregate_image_prediction(
        self,
        candidate_predictions: tuple[Point1Prediction, ...],
    ) -> Point1Prediction:
        """Collapse candidate-level predictions into one image-level Rule 1 decision."""
        selected_prediction = max(
            candidate_predictions,
            key=lambda prediction: (
                _decision_priority(prediction.decision_state),
                prediction.confidence,
            ),
        )
        return Point1Prediction(
            rule_id=selected_prediction.rule_id,
            decision_state=selected_prediction.decision_state,
            target_bbox=(
                selected_prediction.target_bbox
                if selected_prediction.decision_state == "violation"
                else None
            ),
            supporting_evidence_ids=selected_prediction.supporting_evidence_ids,
            counter_evidence_ids=selected_prediction.counter_evidence_ids,
            unknown_items=selected_prediction.unknown_items,
            reason_slots=dict(selected_prediction.reason_slots),
            reason_text=selected_prediction.reason_text,
            confidence=selected_prediction.confidence,
        )


def _decision_priority(decision_state: str) -> int:
    priority = {
        "no_violation": 0,
        "unknown": 1,
        "violation": 2,
    }
    return priority.get(decision_state, -1)


def _cap_candidates(
    candidates: tuple,
    *,
    max_candidates_per_image: int | None,
):
    if max_candidates_per_image is None or len(candidates) <= max_candidates_per_image:
        return candidates
    sorted_candidates = tuple(
        sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    )
    return sorted_candidates[:max_candidates_per_image]
