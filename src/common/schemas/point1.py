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
