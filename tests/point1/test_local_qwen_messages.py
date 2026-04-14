"""Tests for the local Qwen3-VL message builder."""

from __future__ import annotations

from base64 import b64decode
from typing import Any

from benchmark.constructionsite10k.parser import parse_sample
from point1.baselines.local_qwen import LocalQwen3VLClient, LocalQwenLoadConfig


def _valid_png_bytes() -> bytes:
    """Return a tiny valid PNG payload for message-construction tests."""
    return b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aF9sAAAAASUVORK5CYII="
    )


def test_local_qwen_message_builder_supports_five_shot(
    sample_annotation: dict[str, object],
) -> None:
    """Local Qwen message construction should mirror direct/five-shot baseline intent."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": _valid_png_bytes(), "path": "demo.png"},
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


def test_local_qwen_few_shot_assistant_message_uses_text_blocks(
    sample_annotation: dict[str, object],
) -> None:
    """Few-shot assistant examples should use content blocks, not raw strings."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": _valid_png_bytes(), "path": "demo.png"},
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

    assistant_message = messages[2]
    assert isinstance(assistant_message["content"], list)
    assert assistant_message["content"][0]["type"] == "text"


def test_local_qwen_system_message_uses_text_blocks_for_author_vqa(
    sample_annotation: dict[str, object],
) -> None:
    """System messages should use text blocks so Qwen chat templating accepts them."""
    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": _valid_png_bytes(), "path": "demo.png"},
        }
    )
    client = LocalQwen3VLClient(LocalQwenLoadConfig(model_path="demo-model"))

    try:
        messages = client._build_qwen_messages(  # noqa: SLF001
            target_sample=sample,
            mode="direct",
            example_samples=(),
            task_profile="classification_only",
            prompt_style="author_vqa",
        )
    except ImportError:
        return

    system_message = messages[0]
    assert system_message["role"] == "system"
    assert isinstance(system_message["content"], list)
    assert system_message["content"][0]["type"] == "text"


def test_local_qwen_complete_supports_generic_messages() -> None:
    """The generic completion helper should reuse chat templating and strip token_type_ids."""

    class FakeTensor:
        def __init__(self, label: str) -> None:
            self.label = label

        def to(self, device: str) -> FakeTensor:
            return FakeTensor(f"{self.label}@{device}")

    captured: dict[str, Any] = {}

    class FakeProcessor:
        def apply_chat_template(  # noqa: ANN001
            self,
            messages,
            tokenize,
            add_generation_prompt,
            return_dict,
            return_tensors,
        ):
            captured["messages"] = messages
            captured["apply_kwargs"] = {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "return_dict": return_dict,
                "return_tensors": return_tensors,
            }
            return {
                "input_ids": [[10, 11, 12]],
                "pixel_values": FakeTensor("pixels"),
                "token_type_ids": FakeTensor("token-types"),
            }

        def batch_decode(  # noqa: ANN001
            self,
            generated_ids_trimmed,
            skip_special_tokens,
            clean_up_tokenization_spaces,
        ):
            captured["generated_ids_trimmed"] = generated_ids_trimmed
            captured["decode_kwargs"] = {
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
            return ["decoded response"]

    class FakeModel:
        device = "cuda:0"

        def generate(self, **inputs):  # noqa: ANN003
            captured["generate_inputs"] = inputs
            return [[10, 11, 12, 13, 14]]

    client = LocalQwen3VLClient(LocalQwenLoadConfig(model_path="demo-model"))
    client._processor = FakeProcessor()  # noqa: SLF001
    client._model = FakeModel()  # noqa: SLF001

    response = client.complete(
        messages=[{"role": "user", "content": [{"type": "text", "text": "demo"}]}]
    )

    assert response == "decoded response"
    assert captured["messages"][0]["role"] == "user"
    assert "token_type_ids" not in captured["generate_inputs"]
    assert captured["generated_ids_trimmed"] == [[13, 14]]
