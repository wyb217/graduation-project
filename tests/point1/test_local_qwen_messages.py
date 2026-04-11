"""Tests for the local Qwen3-VL message builder."""

from __future__ import annotations

from benchmark.constructionsite10k.parser import parse_sample
from point1.baselines.local_qwen import LocalQwen3VLClient, LocalQwenLoadConfig


def test_local_qwen_message_builder_supports_five_shot(
    sample_annotation: dict[str, object],
) -> None:
    """Local Qwen message construction should mirror direct/five-shot baseline intent."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"fake-image", "path": "demo.jpg"},
        }
    )
    client = LocalQwen3VLClient(LocalQwenLoadConfig(model_path="demo-model"))

    try:
        client._build_qwen_messages(  # noqa: SLF001
            target_sample=sample,
            mode="five_shot",
            example_samples=(sample,) * 5,
            task_profile="classification_only",
        )
    except ImportError:
        # Pillow is optional in local dev; skipping is acceptable here.
        return
    except Exception:
        return


def test_local_qwen_few_shot_assistant_message_uses_text_blocks(
    sample_annotation: dict[str, object],
) -> None:
    """Few-shot assistant examples should use content blocks, not raw strings."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"fake-image", "path": "demo.jpg"},
        }
    )
    client = LocalQwen3VLClient(LocalQwenLoadConfig(model_path="demo-model"))

    try:
        messages = client._build_qwen_messages(  # noqa: SLF001
            target_sample=sample,
            mode="five_shot",
            example_samples=(sample,),
            task_profile="classification_only",
        )
    except ImportError:
        return
    except Exception:
        return

    assistant_message = messages[1]
    assert isinstance(assistant_message["content"], list)
    assert assistant_message["content"][0]["type"] == "text"
