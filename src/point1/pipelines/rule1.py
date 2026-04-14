"""Rule 1 pipeline: person candidates -> predicates -> executor -> explanation."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.schemas.point1 import Point1Prediction
from point1.candidates.person import OpenCVHogPersonCandidateGenerator
from point1.executor.rule1 import execute_rule1_candidate
from point1.predicates.rule1 import HeuristicRule1PredicateExtractor


@dataclass(frozen=True, slots=True)
class Rule1PipelineResult:
    """Image-level Rule 1 pipeline output."""

    image_id: str
    predictions: tuple[Point1Prediction, ...]


class Rule1Pipeline:
    """Run Rule 1 over one image with a detector and predicate extractor."""

    def __init__(
        self,
        *,
        candidate_generator: OpenCVHogPersonCandidateGenerator | object | None = None,
        predicate_extractor: HeuristicRule1PredicateExtractor | object | None = None,
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

    def run(self, sample: ConstructionSiteSample) -> Rule1PipelineResult:
        """Return Rule 1 predictions for one benchmark sample."""
        candidates = self._candidate_generator.generate(sample)
        if not candidates:
            return Rule1PipelineResult(
                image_id=sample.image_id,
                predictions=(self._build_detection_unknown_prediction(),),
            )

        predictions = tuple(
            execute_rule1_candidate(
                candidate,
                self._predicate_extractor.extract(sample, candidate),
            )
            for candidate in candidates
        )
        return Rule1PipelineResult(image_id=sample.image_id, predictions=predictions)

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
