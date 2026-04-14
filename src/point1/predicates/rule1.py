"""Rule 1 predicate extraction from person candidates."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Literal

from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.schemas.bbox import NormalizedBBox
from point1.candidates.person import PersonCandidate

Rule1PredicateState = Literal["yes", "no", "unknown"]


@dataclass(frozen=True, slots=True)
class Rule1PredicateResult:
    """One predicate decision for a Rule 1 person candidate."""

    state: Rule1PredicateState
    score: float
    reason: str
    evidence_bbox: NormalizedBBox | None


@dataclass(frozen=True, slots=True)
class Rule1PredicateSet:
    """The predicate bundle consumed by the Rule 1 executor."""

    candidate_id: str
    person_visible: Rule1PredicateResult
    hard_hat_visible: Rule1PredicateResult
    upper_body_covered: Rule1PredicateResult
    lower_body_covered: Rule1PredicateResult
    toe_covered: Rule1PredicateResult
    ppe_applicable: Rule1PredicateResult = field(
        default_factory=lambda: Rule1PredicateResult(
            state="yes",
            score=1.0,
            reason="Default Rule 1 path assumes the visible worker is on foot.",
            evidence_bbox=None,
        )
    )
    head_region_visible: Rule1PredicateResult = field(
        default_factory=lambda: Rule1PredicateResult(
            state="yes",
            score=1.0,
            reason="Default Rule 1 path assumes the head region is visible.",
            evidence_bbox=None,
        )
    )


class HeuristicRule1PredicateExtractor:
    """Extract Rule 1 predicates with lightweight crop heuristics."""

    def __init__(
        self,
        *,
        min_candidate_width: int = 24,
        min_candidate_height: int = 48,
        hard_hat_ratio_threshold: float = 0.04,
        uncovered_skin_ratio_threshold: float = 0.18,
        toe_skin_ratio_threshold: float = 0.08,
    ) -> None:
        self._min_candidate_width = min_candidate_width
        self._min_candidate_height = min_candidate_height
        self._hard_hat_ratio_threshold = hard_hat_ratio_threshold
        self._uncovered_skin_ratio_threshold = uncovered_skin_ratio_threshold
        self._toe_skin_ratio_threshold = toe_skin_ratio_threshold

    def extract(
        self,
        sample: ConstructionSiteSample,
        candidate: PersonCandidate,
    ) -> Rule1PredicateSet:
        """Return the Rule 1 predicate bundle for one person candidate."""
        image = _load_pil_image(sample)
        candidate_crop = _crop_bbox(image, candidate.bbox)
        crop_width, crop_height = candidate_crop.size
        person_visible = self._detect_person_visibility(candidate, crop_width, crop_height)
        if person_visible.state == "unknown":
            unknown_predicate = Rule1PredicateResult(
                state="unknown",
                score=0.0,
                reason="The person crop is too small or unclear for reliable PPE judgment.",
                evidence_bbox=candidate.bbox,
            )
            return Rule1PredicateSet(
                candidate_id=candidate.candidate_id,
                person_visible=person_visible,
                ppe_applicable=unknown_predicate,
                head_region_visible=unknown_predicate,
                hard_hat_visible=unknown_predicate,
                upper_body_covered=unknown_predicate,
                lower_body_covered=unknown_predicate,
                toe_covered=unknown_predicate,
            )

        head_crop = _slice_vertical_region(candidate_crop, 0.0, 0.25)
        upper_body_crop = _slice_vertical_region(candidate_crop, 0.25, 0.6)
        lower_body_crop = _slice_vertical_region(candidate_crop, 0.6, 0.9)
        foot_crop = _slice_vertical_region(candidate_crop, 0.85, 1.0)

        hard_hat_ratio = _hard_hat_color_ratio(head_crop)
        upper_skin_ratio = _skin_ratio(upper_body_crop)
        lower_skin_ratio = _skin_ratio(lower_body_crop)
        toe_skin_ratio = _skin_ratio(foot_crop)

        return Rule1PredicateSet(
            candidate_id=candidate.candidate_id,
            person_visible=person_visible,
            ppe_applicable=Rule1PredicateResult(
                state="yes",
                score=person_visible.score,
                reason="The visible person candidate is treated as an on-foot Rule 1 subject.",
                evidence_bbox=candidate.bbox,
            ),
            head_region_visible=Rule1PredicateResult(
                state="yes",
                score=person_visible.score,
                reason="The person crop is large enough to inspect the head region.",
                evidence_bbox=candidate.bbox,
            ),
            hard_hat_visible=self._build_hard_hat_result(candidate, hard_hat_ratio),
            upper_body_covered=self._build_body_coverage_result(
                candidate,
                upper_skin_ratio,
                part_name="upper body",
            ),
            lower_body_covered=self._build_body_coverage_result(
                candidate,
                lower_skin_ratio,
                part_name="lower body",
            ),
            toe_covered=self._build_toe_coverage_result(candidate, toe_skin_ratio),
        )

    def _detect_person_visibility(
        self,
        candidate: PersonCandidate,
        crop_width: int,
        crop_height: int,
    ) -> Rule1PredicateResult:
        if crop_width < self._min_candidate_width or crop_height < self._min_candidate_height:
            return Rule1PredicateResult(
                state="unknown",
                score=0.0,
                reason="The person crop is too small for reliable PPE inspection.",
                evidence_bbox=candidate.bbox,
            )
        return Rule1PredicateResult(
            state="yes",
            score=max(0.0, min(1.0, candidate.score)),
            reason="The person candidate is large enough for PPE inspection.",
            evidence_bbox=candidate.bbox,
        )

    def _build_hard_hat_result(
        self,
        candidate: PersonCandidate,
        hard_hat_ratio: float,
    ) -> Rule1PredicateResult:
        if hard_hat_ratio >= self._hard_hat_ratio_threshold:
            return Rule1PredicateResult(
                state="yes",
                score=min(1.0, hard_hat_ratio / max(self._hard_hat_ratio_threshold, 1e-6)),
                reason="Helmet-like colors are visible around the head region.",
                evidence_bbox=candidate.bbox,
            )
        return Rule1PredicateResult(
            state="no",
            score=1.0 - min(1.0, hard_hat_ratio / max(self._hard_hat_ratio_threshold, 1e-6)),
            reason="No helmet-like colors are visible around the head region.",
            evidence_bbox=candidate.bbox,
        )

    def _build_body_coverage_result(
        self,
        candidate: PersonCandidate,
        skin_ratio: float,
        *,
        part_name: str,
    ) -> Rule1PredicateResult:
        if skin_ratio > self._uncovered_skin_ratio_threshold:
            return Rule1PredicateResult(
                state="no",
                score=min(1.0, skin_ratio),
                reason=f"Visible skin ratio suggests the {part_name} may be uncovered.",
                evidence_bbox=candidate.bbox,
            )
        return Rule1PredicateResult(
            state="yes",
            score=1.0 - min(1.0, skin_ratio),
            reason=f"The {part_name} appears covered by work clothing.",
            evidence_bbox=candidate.bbox,
        )

    def _build_toe_coverage_result(
        self,
        candidate: PersonCandidate,
        skin_ratio: float,
    ) -> Rule1PredicateResult:
        if skin_ratio > self._toe_skin_ratio_threshold:
            return Rule1PredicateResult(
                state="no",
                score=min(1.0, skin_ratio),
                reason="Visible skin near the feet suggests shoes may not cover the toes.",
                evidence_bbox=candidate.bbox,
            )
        return Rule1PredicateResult(
            state="yes",
            score=1.0 - min(1.0, skin_ratio),
            reason="Foot region appears consistent with toe-covered footwear.",
            evidence_bbox=candidate.bbox,
        )


def _load_pil_image(sample: ConstructionSiteSample):
    if sample.image is None or sample.image.bytes is None:
        raise ValueError(f"Sample {sample.image_id} does not contain embedded image bytes.")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime dependency only
        raise ImportError("Rule 1 predicate extraction requires Pillow.") from exc
    return Image.open(io.BytesIO(sample.image.bytes)).convert("RGB")


def _crop_bbox(image, bbox: NormalizedBBox):
    image_width, image_height = image.size
    left = int(bbox.x_min * image_width)
    top = int(bbox.y_min * image_height)
    right = max(left + 1, int(bbox.x_max * image_width))
    bottom = max(top + 1, int(bbox.y_max * image_height))
    return image.crop((left, top, right, bottom))


def _slice_vertical_region(image, start_ratio: float, end_ratio: float):
    image_width, image_height = image.size
    top = int(start_ratio * image_height)
    bottom = max(top + 1, int(end_ratio * image_height))
    return image.crop((0, top, image_width, bottom))


def _hard_hat_color_ratio(image) -> float:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime dependency only
        raise ImportError("Rule 1 predicate extraction requires numpy.") from exc

    rgb = np.asarray(image)
    if rgb.size == 0:
        return 0.0
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    yellow_orange = (red > 150) & (green > 100) & (blue < 150)
    white = (red > 190) & (green > 190) & (blue > 190)
    red_hat = (red > 150) & (green < 120) & (blue < 120)
    hard_hat_like = yellow_orange | white | red_hat
    return float(hard_hat_like.mean())


def _skin_ratio(image) -> float:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime dependency only
        raise ImportError("Rule 1 predicate extraction requires numpy.") from exc

    rgb = np.asarray(image)
    if rgb.size == 0:
        return 0.0
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    skin_mask = (
        (red > 95)
        & (green > 40)
        & (blue > 20)
        & ((max_rgb - min_rgb) > 15)
        & (abs(red - green) > 15)
        & (red > green)
        & (red > blue)
    )
    return float(skin_mask.mean())
