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
All bbox coordinates must be floats between 0.0 and 1.0.
Do not return pixel coordinates such as 450 or 1024.
Use empty arrays for supporting_evidence_ids and counter_evidence_ids in this baseline.
Keep explanations short and factual.
""".strip()


STRUCTURED_TASK_PROMPT = """
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


CLASSIFICATION_ONLY_TASK_PROMPT = """
Inspect the image and return the closed-domain Point 1 JSON object.
This run is classification-only. Do not localize boxes.
Always set target_bbox to null for all four rules.
Return exactly four predictions, one per rule.
{
  "image_id": "...",
  "predictions": [
    {
      "rule_id": 1,
      "decision_state": "violation | no_violation | unknown",
      "target_bbox": null,
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


AUTHOR_STYLE_FIVE_SHOT_STRUCTURED_PROMPT = """
You are a construction site safety inspector.
Review the image according to the following four safety rules:
1. Basic PPE
2. Safety harness when working at height without protection
3. Edge protection / warning for underground projects
4. Workers inside excavator blind spots or operating radius

Return JSON only in this format:
{
  "image_id": "...",
  "violated_rule_ids": [1, 3],
  "explanation": "...",
  "target_bbox": [x_min, y_min, x_max, y_max] or null
}

Requirements:
- `violated_rule_ids` is a list of violated rule IDs.
- Use an empty list if no visible rule is violated.
- Give one short explanation covering the visible violation(s).
- `target_bbox` should ground the main visible violation.
- Use normalized coordinates between 0.0 and 1.0.
""".strip()


AUTHOR_STYLE_FIVE_SHOT_CLASSIFICATION_PROMPT = """
You are a construction site safety inspector.
Review the image according to the following four safety rules:
1. Basic PPE
2. Safety harness when working at height without protection
3. Edge protection / warning for underground projects
4. Workers inside excavator blind spots or operating radius

Return JSON only in this format:
{
  "image_id": "...",
  "violated_rule_ids": [1, 3],
  "explanation": "...",
  "target_bbox": null
}

Requirements:
- `violated_rule_ids` is a list of violated rule IDs.
- Use an empty list if no visible rule is violated.
- Give one short explanation covering the visible violation(s).
- Always set `target_bbox` to null.
""".strip()


AUTHOR_VQA_SYSTEM_PROMPT = """
You are a construction site safety inspector.
You are responsible for viewing the given image and give helpful,
detailed, and polite answers to your supervisor.
You only answer questions that are asked by the supervisor and in the
exact way as requested.
""".strip()


AUTHOR_VQA_FEW_SHOT_PREAMBLE = """
You will be asked to read the image and identify violations of safety
rules that appears in the image.
You also need to provide a short reasoning and bounding boxs showing the
location of the violation.

I will give you five examples to show you how your answer should be
formatted and what your reasoning should include.
""".strip()


AUTHOR_VQA_STRUCTURED_PROMPT = """
Please read the image and identify if there are violations of the
following four safety rules in the image.
Do not include violations that do not exist in your answer.
Assume no violation if the visual information is not enough to make a
judgement:

1. Use of basic PPE when on foot at construction sites. Machine
operators do not need PPE. (hard hats, properly worn clothes covering
shoulders and legs, shoes that can cover toes, high-visibility
retroreflective vests at night, face shield or safety glasses when
cutting, welding, grinding, or drilling).

2. Use of safety harness when working from a height of three meters and
the edges are without any edge protection.

3. Adoption of edge protection or edge warning including guardrails,
fences, for underground projects three meters in depth with steep
retaining wall and for human to stand.

4. Appearance of worker in the blind spots of the operator and within
the operation radius of excavators in operation, or excavators with
operators inside.

Your answer should be in the format of:
{"id of the safety rule": {"reason": one or two sentences explaining who
violate the rule in the image and the specific reason, "bounding_box":
[the location of violation in the image x_min, y_min, x_max, y_max in
0-1 normalized space]}}.

Return {"0": "No violations"} if you find no violation in the image.
""".strip()


AUTHOR_VQA_CLASSIFICATION_PROMPT = """
Please read the image and identify if there are violations of the
following four safety rules in the image.
Do not include violations that do not exist in your answer.
Assume no violation if the visual information is not enough to make a
judgement:

1. Use of basic PPE when on foot at construction sites. Machine
operators do not need PPE. (hard hats, properly worn clothes covering
shoulders and legs, shoes that can cover toes, high-visibility
retroreflective vests at night, face shield or safety glasses when
cutting, welding, grinding, or drilling).

2. Use of safety harness when working from a height of three meters and
the edges are without any edge protection.

3. Adoption of edge protection or edge warning including guardrails,
fences, for underground projects three meters in depth with steep
retaining wall and for human to stand.

4. Appearance of worker in the blind spots of the operator and within
the operation radius of excavators in operation, or excavators with
operators inside.

This run is classification-only.
Always set "bounding_box" to null whenever you report a violation.
Your answer should be in the format of:
{"id of the safety rule": {"reason": one or two sentences explaining who
violate the rule in the image and the specific reason, "bounding_box":
null}}.

Return {"0": "No violations"} if you find no violation in the image.
""".strip()


AUTHOR_TRAIN_MIMIC_FIVE_SHOT_IDS = (
    "0004852",  # clean
    "0005167",  # rule1 + rule3
    "0004858",  # rule1
    "0004850",  # rule4
    "0005509",  # rule1 + rule2
)


def build_example_prediction_set(
    sample: ConstructionSiteSample,
    *,
    task_profile: str = "structured",
) -> Point1ImagePredictionSet:
    """Convert one benchmark sample into the structured output used in 5-shot examples."""
    predictions: list[Point1Prediction] = []
    include_bbox = task_profile == "structured"
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

        target_bbox = (
            violation.bounding_boxes[0] if include_bbox and violation.bounding_boxes else None
        )
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


def select_five_shot_ids(
    registry: SplitRegistry | None,
    *,
    subset_name: str = "balanced_dev_15x5",
    example_profile: str = "balanced_single_rule",
    explicit_image_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Select stable few-shot example IDs for the requested profile."""
    if explicit_image_ids:
        return explicit_image_ids
    if example_profile == "balanced_single_rule":
        if registry is None:
            raise ValueError("balanced_single_rule examples require a split registry.")
        return select_default_five_shot_ids(registry, subset_name=subset_name)
    if example_profile == "author_train_mimic":
        return AUTHOR_TRAIN_MIMIC_FIVE_SHOT_IDS
    raise ValueError(f"Unknown example_profile: {example_profile}")


def build_inference_messages(
    *,
    target_sample: ConstructionSiteSample,
    mode: str,
    example_samples: tuple[ConstructionSiteSample, ...],
    task_profile: str = "structured",
    prompt_style: str = "default",
) -> list[dict[str, object]]:
    """Build the multimodal message list for one target image."""
    messages: list[dict[str, object]] = [
        {"role": "system", "content": get_system_prompt(prompt_style)}
    ]
    if mode == "five_shot":
        if prompt_style == "author_vqa":
            messages.append(
                {
                    "role": "user",
                    "content": AUTHOR_VQA_FEW_SHOT_PREAMBLE,
                }
            )
        for example_sample in example_samples:
            messages.append(
                _build_five_shot_user_message(
                    example_sample,
                    task_profile=task_profile,
                    prompt_style=prompt_style,
                )
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        build_example_answer(
                            example_sample,
                            task_profile=task_profile,
                            prompt_style=prompt_style,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                }
            )
        messages.append(
            _build_five_shot_user_message(
                target_sample,
                task_profile=task_profile,
                prompt_style=prompt_style,
            )
        )
        return messages
    messages.append(
        _build_user_message(
            target_sample,
            task_profile=task_profile,
            prompt_style=prompt_style,
        )
    )
    return messages


def _build_user_message(
    sample: ConstructionSiteSample,
    *,
    task_profile: str,
    prompt_style: str,
) -> dict[str, object]:
    if sample.image is None or sample.image.bytes is None:
        raise ValueError(f"Sample {sample.image_id} does not contain embedded image bytes.")
    image_base64 = base64.b64encode(sample.image.bytes).decode("utf-8")
    task_prompt = get_task_prompt(task_profile, prompt_style=prompt_style)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{task_prompt}\nImage ID: {sample.image_id}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        ],
    }


def _build_five_shot_user_message(
    sample: ConstructionSiteSample,
    *,
    task_profile: str,
    prompt_style: str,
) -> dict[str, object]:
    if sample.image is None or sample.image.bytes is None:
        raise ValueError(f"Sample {sample.image_id} does not contain embedded image bytes.")
    image_base64 = base64.b64encode(sample.image.bytes).decode("utf-8")
    task_prompt = get_five_shot_task_prompt(task_profile, prompt_style=prompt_style)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{task_prompt}\nImage ID: {sample.image_id}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        ],
    }


def get_system_prompt(prompt_style: str) -> str:
    """Return the system prompt for the selected prompt style."""
    if prompt_style == "default":
        return SYSTEM_PROMPT
    if prompt_style == "author_vqa":
        return AUTHOR_VQA_SYSTEM_PROMPT
    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def get_task_prompt(task_profile: str, *, prompt_style: str = "default") -> str:
    """Return the direct-prompt task instruction for the selected prompt style."""
    if prompt_style == "author_vqa":
        if task_profile == "structured":
            return AUTHOR_VQA_STRUCTURED_PROMPT
        if task_profile == "classification_only":
            return AUTHOR_VQA_CLASSIFICATION_PROMPT
        raise ValueError(f"Unknown task_profile: {task_profile}")
    if task_profile == "structured":
        return STRUCTURED_TASK_PROMPT
    if task_profile == "classification_only":
        return CLASSIFICATION_ONLY_TASK_PROMPT
    raise ValueError(f"Unknown task_profile: {task_profile}")


def get_five_shot_task_prompt(task_profile: str, *, prompt_style: str = "default") -> str:
    """Return the task prompt used for five-shot VQA."""
    if prompt_style == "author_vqa":
        return get_task_prompt(task_profile, prompt_style=prompt_style)
    if task_profile == "structured":
        return AUTHOR_STYLE_FIVE_SHOT_STRUCTURED_PROMPT
    if task_profile == "classification_only":
        return AUTHOR_STYLE_FIVE_SHOT_CLASSIFICATION_PROMPT
    raise ValueError(f"Unknown task_profile: {task_profile}")


def build_author_style_example_answer(
    sample: ConstructionSiteSample,
    *,
    task_profile: str,
) -> dict[str, object]:
    """Build an author-style few-shot answer with violated rule IDs and one explanation."""
    violated_rule_ids = [
        rule_id for rule_id, violation in sorted(sample.violations.items()) if violation is not None
    ]
    explanation_parts = [
        violation.reason
        for _, violation in sorted(sample.violations.items())
        if violation is not None and violation.reason
    ]
    explanation = (
        "No visible safety violation." if not explanation_parts else " ".join(explanation_parts)
    )
    target_bbox = None
    if task_profile == "structured":
        for _, violation in sorted(sample.violations.items()):
            if violation is not None and violation.bounding_boxes:
                target_bbox = violation.bounding_boxes[0].to_list()
                break
    return {
        "image_id": sample.image_id,
        "violated_rule_ids": violated_rule_ids,
        "explanation": explanation,
        "target_bbox": target_bbox,
    }


def build_author_vqa_example_answer(
    sample: ConstructionSiteSample,
    *,
    task_profile: str,
) -> dict[str, object]:
    """Build a sparse rule dictionary answer matching the official author prompt style."""
    answer: dict[str, object] = {"image_id": sample.image_id}
    for rule_id, violation in sorted(sample.violations.items()):
        if violation is None:
            continue
        answer[str(rule_id)] = {
            "reason": violation.reason,
            "bounding_box": (
                violation.bounding_boxes[0].to_list()
                if task_profile == "structured" and violation.bounding_boxes
                else None
            ),
        }
    if len(answer) == 1:
        answer["0"] = "No violations"
    return answer


def build_example_answer(
    sample: ConstructionSiteSample,
    *,
    task_profile: str,
    prompt_style: str,
) -> dict[str, object]:
    """Build the example answer matching the selected prompt style."""
    if prompt_style == "author_vqa":
        return build_author_vqa_example_answer(sample, task_profile=task_profile)
    return build_author_style_example_answer(sample, task_profile=task_profile)
