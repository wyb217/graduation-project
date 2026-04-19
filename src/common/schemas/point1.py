"""Structured Point 1 output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from common.schemas.bbox import NormalizedBBox


@dataclass(frozen=True, slots=True)
class Point1Evidence:
    """A single evidence item consumed by the Point 1 rule executor."""

    evidence_id: str
    kind: str
    bbox: NormalizedBBox | None = None
    score: float | None = None
    attributes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "bbox": None if self.bbox is None else self.bbox.to_list(),
            "score": self.score,
            "attributes": self.attributes,
        }


@dataclass(frozen=True, slots=True)
class Point1Prediction:
    """A structured Point 1 decision payload."""

    rule_id: int
    decision_state: str
    target_bbox: NormalizedBBox | None
    supporting_evidence_ids: tuple[str, ...]
    counter_evidence_ids: tuple[str, ...]
    unknown_items: tuple[str, ...]
    reason_slots: dict[str, str]
    reason_text: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "rule_id": self.rule_id,
            "decision_state": self.decision_state,
            "target_bbox": None if self.target_bbox is None else self.target_bbox.to_list(),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "counter_evidence_ids": list(self.counter_evidence_ids),
            "unknown_items": list(self.unknown_items),
            "reason_slots": self.reason_slots,
            "reason_text": self.reason_text,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class Point1ImagePredictionSet:
    """The four-rule structured output for a single image."""

    image_id: str
    predictions: tuple[Point1Prediction, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "image_id": self.image_id,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


@dataclass(frozen=True, slots=True)
class Point1BaselineRecord:
    """A stored baseline inference record for one image."""

    image_id: str
    provider_name: str
    model_name: str
    mode: str
    raw_response_text: str
    parsed_output: Point1ImagePredictionSet | None
    error_message: str | None = None
    candidate_ms: float | None = None
    predicate_ms: float | None = None
    executor_ms: float | None = None
    total_ms: float | None = None
    candidate_count: int | None = None
    candidate_count_raw: int | None = None
    candidate_count_capped: int | None = None
    fallback_used: bool | None = None
    predicate_backend: str | None = None
    candidate_batch_size: int | None = None
    max_new_tokens: int | None = None
    max_candidates_per_image: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "image_id": self.image_id,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "mode": self.mode,
            "raw_response_text": self.raw_response_text,
            "parsed_output": None if self.parsed_output is None else self.parsed_output.to_dict(),
            "error_message": self.error_message,
            "candidate_ms": self.candidate_ms,
            "predicate_ms": self.predicate_ms,
            "executor_ms": self.executor_ms,
            "total_ms": self.total_ms,
            "candidate_count": self.candidate_count,
            "candidate_count_raw": self.candidate_count_raw,
            "candidate_count_capped": self.candidate_count_capped,
            "fallback_used": self.fallback_used,
            "predicate_backend": self.predicate_backend,
            "candidate_batch_size": self.candidate_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "max_candidates_per_image": self.max_candidates_per_image,
        }
