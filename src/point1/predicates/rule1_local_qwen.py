"""Local-Qwen-backed Rule 1 predicate extraction over person crops."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from typing import Any

from benchmark.constructionsite10k.types import ConstructionSiteSample
from point1.baselines.local_qwen import LocalQwen3VLClient
from point1.candidates.person import PersonCandidate
from point1.predicates.rule1 import Rule1PredicateResult, Rule1PredicateSet
from point1.predicates.rule1_vlm import RULE1_VLM_SYSTEM_PROMPT, RULE1_VLM_USER_PROMPT

VALID_STATES = {"yes", "no", "unknown"}


@dataclass(frozen=True, slots=True)
class LocalQwenRule1PredicateExtractor:
    """Extract Rule 1 predicates from a person crop with a local Qwen3-VL model."""

    client: LocalQwen3VLClient
    model_name: str
    min_candidate_width: int = 24
    min_candidate_height: int = 48

    def extract(
        self,
        sample: ConstructionSiteSample,
        candidate: PersonCandidate,
    ) -> Rule1PredicateSet:
        """Return a Rule 1 predicate bundle for one person candidate."""
        image = _load_pil_image(sample)
        candidate_crop = _crop_bbox(image, candidate.bbox)
        crop_width, crop_height = candidate_crop.size

        if crop_width < self.min_candidate_width or crop_height < self.min_candidate_height:
            return _build_unknown_predicate_set(
                candidate=candidate,
                reason="The person crop is too small for reliable local-Qwen PPE inspection.",
            )

        raw_response = self.client.complete(
            messages=_build_messages(candidate_crop, candidate_id=candidate.candidate_id)
        )
        payload = _load_json_payload(raw_response)
        return Rule1PredicateSet(
            candidate_id=candidate.candidate_id,
            person_visible=_parse_predicate_result(
                payload=payload,
                field_name="person_visible",
                candidate=candidate,
            ),
            ppe_applicable=_parse_predicate_result(
                payload=payload,
                field_name="ppe_applicable",
                candidate=candidate,
            ),
            head_region_visible=_parse_predicate_result(
                payload=payload,
                field_name="head_region_visible",
                candidate=candidate,
            ),
            hard_hat_visible=_parse_predicate_result(
                payload=payload,
                field_name="hard_hat_visible",
                candidate=candidate,
            ),
            upper_body_covered=_parse_predicate_result(
                payload=payload,
                field_name="upper_body_covered",
                candidate=candidate,
            ),
            lower_body_covered=_parse_predicate_result(
                payload=payload,
                field_name="lower_body_covered",
                candidate=candidate,
            ),
            toe_covered=_parse_predicate_result(
                payload=payload,
                field_name="toe_covered",
                candidate=candidate,
            ),
        )


def _build_messages(image, *, candidate_id: str) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": RULE1_VLM_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{RULE1_VLM_USER_PROMPT}\nCandidate ID: {candidate_id}"},
            ],
        },
    ]


def _build_unknown_predicate_set(
    *,
    candidate: PersonCandidate,
    reason: str,
) -> Rule1PredicateSet:
    unknown_result = Rule1PredicateResult(
        state="unknown",
        score=0.0,
        reason=reason,
        evidence_bbox=candidate.bbox,
    )
    return Rule1PredicateSet(
        candidate_id=candidate.candidate_id,
        person_visible=unknown_result,
        hard_hat_visible=unknown_result,
        upper_body_covered=unknown_result,
        lower_body_covered=unknown_result,
        toe_covered=unknown_result,
        ppe_applicable=unknown_result,
        head_region_visible=unknown_result,
    )


def _parse_predicate_result(
    *,
    payload: dict[str, Any],
    field_name: str,
    candidate: PersonCandidate,
) -> Rule1PredicateResult:
    raw_value = payload.get(field_name)
    if not isinstance(raw_value, dict):
        raise ValueError(f"Missing predicate field: {field_name}")
    state = str(raw_value.get("state", "")).strip().lower()
    if state not in VALID_STATES:
        raise ValueError(f"Invalid predicate state for {field_name}: {state!r}")
    score = float(raw_value.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    return Rule1PredicateResult(
        state=state,
        score=score,
        reason=str(raw_value.get("reason", "")).strip(),
        evidence_bbox=candidate.bbox,
    )


def _load_json_payload(response_text: str) -> dict[str, Any]:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
        if match is None:
            raise ValueError("Could not find JSON object inside markdown fence.")
        stripped = match.group(1)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Rule 1 predicate response must be a JSON object.")
    return payload


def _load_pil_image(sample: ConstructionSiteSample):
    if sample.image is None or sample.image.bytes is None:
        raise ValueError(f"Sample {sample.image_id} does not contain embedded image bytes.")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime dependency only
        raise ImportError("Local-Qwen Rule 1 predicate extraction requires Pillow.") from exc
    return Image.open(io.BytesIO(sample.image.bytes)).convert("RGB")


def _crop_bbox(image, bbox):
    image_width, image_height = image.size
    left = int(bbox.x_min * image_width)
    top = int(bbox.y_min * image_height)
    right = max(left + 1, int(bbox.x_max * image_width))
    bottom = max(top + 1, int(bbox.y_max * image_height))
    return image.crop((left, top, right, bottom))
