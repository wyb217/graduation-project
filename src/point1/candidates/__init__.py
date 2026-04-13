"""Candidate generation utilities for Point 1 pipelines."""

from point1.candidates.person import (
    OpenCVHogPersonCandidateGenerator,
    PersonCandidate,
    pixel_xyxy_to_normalized_bbox,
)

__all__ = [
    "OpenCVHogPersonCandidateGenerator",
    "PersonCandidate",
    "pixel_xyxy_to_normalized_bbox",
]
