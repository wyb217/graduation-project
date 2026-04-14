"""VLM-backed Rule 1 predicate extraction over person crops."""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from typing import Any

from benchmark.constructionsite10k.types import ConstructionSiteSample
from point1.baselines.client import VisionLanguageClient
from point1.candidates.person import PersonCandidate
from point1.predicates.rule1 import (
    Rule1PredicateResult,
    Rule1PredicateSet,
)

RULE1_VLM_SYSTEM_PROMPT = """
You inspect one cropped worker image from a construction site.
Return JSON only. Do not use markdown or extra commentary.
For each predicate, use state in {"yes","no","unknown"}.
Scores must be floats between 0.0 and 1.0.
Be conservative: if the crop is unclear or the body part is not visible, use unknown.
""".strip()

RULE1_VLM_USER_PROMPT = """
Judge the visible worker crop for Rule 1 PPE predicates.
Return exactly this JSON object:
{
  "person_visible": {"state": "yes|no|unknown", "score": 0.0, "reason": "..."},
  "ppe_applicable": {"state": "yes|no|unknown", "score": 0.0, "reason": "..."},
  "head_region_visible": {"state": "yes|no|unknown", "score": 0.0, "reason": "..."},
  "hard_hat_visible": {"state": "yes|no|unknown", "score": 0.0, "reason": "..."},
  "upper_body_covered": {"state": "yes|no|unknown", "score": 0.0, "reason": "..."},
  "lower_body_covered": {"state": "yes|no|unknown", "score": 0.0, "reason": "..."},
  "toe_covered": {"state": "yes|no|unknown", "score": 0.0, "reason": "..."}
}
Definitions:
- person_visible: whether this crop contains a clearly inspectable worker.
- ppe_applicable: whether Rule 1 applies to this candidate as an on-foot worker. Use "no" for
  excavator operators, cab occupants, or non-worker false detections. Use "unknown" if unclear.
- head_region_visible: whether the head/helmet region is actually visible enough to judge.
- hard_hat_visible: whether a hard hat is visible on the head.
- upper_body_covered: whether shoulders/upper body are covered by work clothing.
- lower_body_covered: whether legs/lower body are covered by work clothing.
- toe_covered: whether footwear appears to cover the toes.
Critical constraints:
- This candidate may also violate Rule 2, 3, or 4. Judge only whether Rule 1 applies to this
  candidate and whether the visible PPE items are present.
- Do not set ppe_applicable to no only because excavators, pits, edges, barriers, or other scene
  risks are nearby. An on-foot worker can violate Rule 1 and another rule at the same time.
- Do not infer missing PPE from an invisible body part; use "unknown" instead.
- Do not use excavators, pits, edges, barriers, or other scene risks as evidence of Rule 1 PPE
  violations.
""".strip()

VALID_STATES = {"yes", "no", "unknown"}


@dataclass(frozen=True, slots=True)
class VLMRule1PredicateExtractor:
    """Extract Rule 1 predicates from a person crop with an OpenAI-compatible VLM."""

    client: VisionLanguageClient
    model_name: str
    provider_name: str
    min_candidate_width: int = 24
    min_candidate_height: int = 48
    temperature: float = 0.0
    max_tokens: int = 500

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
                reason="The person crop is too small for reliable VLM PPE inspection.",
            )

        raw_response = self.client.complete(
            messages=_build_messages(candidate_crop, candidate_id=candidate.candidate_id),
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
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
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return [
        {"role": "system", "content": RULE1_VLM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{RULE1_VLM_USER_PROMPT}\nCandidate ID: {candidate_id}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
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
        raise ImportError("VLM Rule 1 predicate extraction requires Pillow.") from exc
    return Image.open(io.BytesIO(sample.image.bytes)).convert("RGB")


def _crop_bbox(image, bbox):
    image_width, image_height = image.size
    left = int(bbox.x_min * image_width)
    top = int(bbox.y_min * image_height)
    right = max(left + 1, int(bbox.x_max * image_width))
    bottom = max(top + 1, int(bbox.y_max * image_height))
    return image.crop((left, top, right, bottom))
