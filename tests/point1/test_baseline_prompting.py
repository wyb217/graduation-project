"""Tests for Point 1 API baseline prompts."""

from __future__ import annotations

from benchmark.constructionsite10k.parser import parse_sample
from benchmark.constructionsite10k.registry import SplitRegistry
from point1.baselines.prompting import (
    build_author_style_example_answer,
    build_example_prediction_set,
    build_inference_messages,
    select_default_five_shot_ids,
)


def test_build_example_prediction_set_contains_four_rule_outputs(
    sample_annotation: dict[str, object],
) -> None:
    """Ground-truth example payloads should cover all four rules."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"demo-image", "path": "0000424.jpg"},
            "rule_2_violation": {"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "rule 2"},
        }
    )

    payload = build_example_prediction_set(sample).to_dict()

    assert payload["image_id"] == "0000424"
    assert len(payload["predictions"]) == 4
    assert payload["predictions"][0]["decision_state"] == "violation"
    assert payload["predictions"][1]["decision_state"] == "violation"


def test_select_default_five_shot_ids_picks_one_example_per_bucket(tmp_path) -> None:
    """The default 5-shot selector should choose one ID from each dev bucket."""
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        """
{
  "balanced_dev_15x5_clean": ["c1", "c2"],
  "balanced_dev_15x5_rule1": ["r1a", "r1b"],
  "balanced_dev_15x5_rule2": ["r2a", "r2b"],
  "balanced_dev_15x5_rule3": ["r3a", "r3b"],
  "balanced_dev_15x5_rule4": ["r4a", "r4b"],
  "balanced_dev_15x5": ["ignored"]
}
        """.strip(),
        encoding="utf-8",
    )
    registry = SplitRegistry.from_json(registry_path)

    shot_ids = select_default_five_shot_ids(registry, subset_name="balanced_dev_15x5")

    assert shot_ids == ("c1", "r1a", "r2a", "r3a", "r4a")


def test_build_inference_messages_adds_five_shot_examples(
    sample_annotation: dict[str, object],
) -> None:
    """Five-shot prompting should add five user/assistant example pairs before the target query."""
    example_sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"example-image", "path": "example.jpg"},
        }
    )
    target_sample = parse_sample(
        {
            **sample_annotation,
            "image_id": "target",
            "image": {"bytes": b"target-image", "path": "target.jpg"},
        }
    )

    messages = build_inference_messages(
        target_sample=target_sample,
        mode="five_shot",
        example_samples=(example_sample,) * 5,
    )

    assert messages[0]["role"] == "system"
    assert len(messages) == 12
    assert messages[-1]["role"] == "user"
    assert "violated_rule_ids" in messages[-1]["content"][0]["text"]
    assert (
        "violated_rule_ids" in messages[1]["content"][0]["text"]
        if isinstance(messages[1]["content"], list)
        else True
    )


def test_build_inference_messages_supports_classification_only_profile(
    sample_annotation: dict[str, object],
) -> None:
    """Classification-only prompting should explicitly force null bbox outputs."""
    target_sample = parse_sample(
        {
            **sample_annotation,
            "image_id": "target",
            "image": {"bytes": b"target-image", "path": "target.jpg"},
        }
    )

    messages = build_inference_messages(
        target_sample=target_sample,
        mode="direct",
        example_samples=(),
        task_profile="classification_only",
    )

    text_block = messages[-1]["content"][0]["text"]
    assert "Always set target_bbox to null" in text_block


def test_build_author_style_example_answer_aggregates_rule_ids(
    sample_annotation: dict[str, object],
) -> None:
    """Author-style examples should aggregate violated rule IDs into one answer object."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"demo-image", "path": "0000424.jpg"},
            "rule_2_violation": {"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "rule 2"},
        }
    )

    answer = build_author_style_example_answer(sample, task_profile="structured")

    assert answer["violated_rule_ids"] == [1, 2]
    assert answer["target_bbox"] == [0.22, 0.59, 0.28, 0.75]
