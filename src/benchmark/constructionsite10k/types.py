"""Strongly typed ConstructionSite10k sample contracts."""

from __future__ import annotations

from dataclasses import dataclass

from common.schemas.bbox import NormalizedBBox


@dataclass(frozen=True, slots=True)
class SampleImage:
    """Embedded image payload metadata from parquet exports."""

    path: str | None
    bytes: bytes | None


@dataclass(frozen=True, slots=True)
class SampleAttributes:
    """Scene-level attributes from ConstructionSite10k annotations."""

    illumination: str
    camera_distance: str
    view: str
    quality_of_info: str


@dataclass(frozen=True, slots=True)
class RuleViolation:
    """A single rule violation entry from the benchmark annotation."""

    rule_id: int
    reason: str
    bounding_boxes: tuple[NormalizedBBox, ...]


@dataclass(frozen=True, slots=True)
class ConstructionSiteSample:
    """A parsed benchmark sample with typed access to rules and object boxes."""

    image_id: str
    image: SampleImage | None
    image_caption: str
    attributes: SampleAttributes
    violations: dict[int, RuleViolation | None]
    object_boxes: dict[str, tuple[NormalizedBBox, ...]]
