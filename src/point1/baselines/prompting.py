"""Prompt construction for direct and 5-shot Point 1 API baselines."""

from __future__ import annotations

import base64
import json

from benchmark.constructionsite10k.registry import SplitRegistry
from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.schemas.point1 import Point1ImagePredictionSet, Point1Prediction

SYSTEM_PROMPT = """
You are evaluating a single construction-site image under exactly four closed-domain safety rules.
Return JSON only. Do not use markdown. Do not add commentary outside the JSON object.
For every image, output exactly four predictions, one for each rule_id in [1, 2, 3, 4].
decision_state must be one of: violation, no_violation, unknown.
If there is no visible violation, use target_bbox as null.
If a violation is visible, provide the primary violating bbox
as normalized [x_min, y_min, x_max, y_max].
Use empty arrays for supporting_evidence_ids and counter_evidence_ids in this baseline.
Keep explanations short and factual.
""".strip()


TASK_PROMPT = """
Inspect the image and return the closed-domain Point 1 JSON object:
{
  "image_id": "...",
  "predictions": [
    {
      "rule_id": 1,
      "decision_state": "violation | no_violation | unknown",
      "target_bbox": [x_min, y_min, x_max, y_max] or null,
      "supporting_evidence_ids": [],
      "counter_evidence_ids": [],
      "unknown_items": [],
      "reason_slots": {},
      "reason_text": "...",
      "confidence": 0.0
    }
  ]
}
Rules:
1. Basic PPE
2. Safety harness when working at height without protection
3. Edge protection / warning for underground projects
4. Workers inside excavator blind spots or operating radius
""".strip()


def build_example_prediction_set(sample: ConstructionSiteSample) -> Point1ImagePredictionSet:
    """Convert one benchmark sample into the structured output used in 5-shot examples."""
    predictions: list[Point1Prediction] = []
    for rule_id in range(1, 5):
        violation = sample.violations.get(rule_id)
        if violation is None:
            predictions.append(
                Point1Prediction(
                    rule_id=rule_id,
                    decision_state="no_violation",
                    target_bbox=None,
                    supporting_evidence_ids=(),
                    counter_evidence_ids=(),
                    unknown_items=(),
                    reason_slots={},
                    reason_text=f"No visible violation for rule {rule_id}.",
                    confidence=1.0,
                )
            )
            continue

        target_bbox = violation.bounding_boxes[0] if violation.bounding_boxes else None
        predictions.append(
            Point1Prediction(
                rule_id=rule_id,
                decision_state="violation",
                target_bbox=target_bbox,
                supporting_evidence_ids=(),
                counter_evidence_ids=(),
                unknown_items=(),
                reason_slots={"source": "ground_truth_example"},
                reason_text=violation.reason,
                confidence=1.0,
            )
        )

    return Point1ImagePredictionSet(image_id=sample.image_id, predictions=tuple(predictions))


def select_default_five_shot_ids(
    registry: SplitRegistry,
    *,
    subset_name: str = "balanced_dev_15x5",
) -> tuple[str, str, str, str, str]:
    """Pick one stable example ID from clean and each single-rule bucket."""
    return (
        registry.get_split(f"{subset_name}_clean")[0],
        registry.get_split(f"{subset_name}_rule1")[0],
        registry.get_split(f"{subset_name}_rule2")[0],
        registry.get_split(f"{subset_name}_rule3")[0],
        registry.get_split(f"{subset_name}_rule4")[0],
    )


def build_inference_messages(
    *,
    target_sample: ConstructionSiteSample,
    mode: str,
    example_samples: tuple[ConstructionSiteSample, ...],
) -> list[dict[str, object]]:
    """Build the multimodal message list for one target image."""
    messages: list[dict[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if mode == "five_shot":
        for example_sample in example_samples:
            messages.append(_build_user_message(example_sample))
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        build_example_prediction_set(example_sample).to_dict(),
                        ensure_ascii=False,
                        indent=2,
                    ),
                }
            )
    messages.append(_build_user_message(target_sample))
    return messages


def _build_user_message(sample: ConstructionSiteSample) -> dict[str, object]:
    if sample.image is None or sample.image.bytes is None:
        raise ValueError(f"Sample {sample.image_id} does not contain embedded image bytes.")
    image_base64 = base64.b64encode(sample.image.bytes).decode("utf-8")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{TASK_PROMPT}\nImage ID: {sample.image_id}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        ],
    }
