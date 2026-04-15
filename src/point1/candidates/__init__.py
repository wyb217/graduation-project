"""Candidate generation utilities for Point 1 pipelines."""

from point1.candidates.person import (
    HogThenTorchvisionPersonCandidateGenerator,
    OpenCVHogPersonCandidateGenerator,
    PersonCandidate,
    TorchvisionPersonCandidateGenerator,
    pixel_xyxy_to_normalized_bbox,
)

__all__ = [
    "HogThenTorchvisionPersonCandidateGenerator",
    "OpenCVHogPersonCandidateGenerator",
    "PersonCandidate",
    "TorchvisionPersonCandidateGenerator",
    "pixel_xyxy_to_normalized_bbox",
]
