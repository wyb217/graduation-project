"""Local Qwen3-VL baseline runner for server-side inference."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.schemas.point1 import Point1BaselineRecord
from point1.baselines.parsing import parse_prediction_set_response
from point1.baselines.prompting import (
    build_author_style_example_answer,
    get_five_shot_task_prompt,
    get_task_prompt,
)


@dataclass(frozen=True, slots=True)
class LocalQwenLoadConfig:
    """Configuration for loading a local Qwen3-VL model."""

    model_path: str
    torch_dtype: str = "auto"
    max_new_tokens: int = 1200
    attn_implementation: str = "sdpa"


class LocalQwen3VLClient:
    """Thin lazy wrapper around a local Qwen3-VL Transformers model."""

    def __init__(self, config: LocalQwenLoadConfig) -> None:
        self._config = config
        self._model = None
        self._processor = None

    def complete_for_sample(
        self,
        *,
        target_sample: ConstructionSiteSample,
        mode: str,
        example_samples: tuple[ConstructionSiteSample, ...],
        task_profile: str,
    ) -> str:
        """Run one local Qwen3-VL generation and return the raw text output."""
        self._ensure_loaded()
        messages = self._build_qwen_messages(
            target_sample=target_sample,
            mode=mode,
            example_samples=example_samples,
            task_profile=task_profile,
        )
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(self._model.device)
        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=self._config.max_new_tokens,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids, strict=True)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - only triggered on missing local deps
            raise ImportError(
                "Local Qwen3-VL inference requires torch, pillow, accelerate, and transformers."
            ) from exc

        dtype = self._config.torch_dtype
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._config.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation=self._config.attn_implementation,
        )
        self._processor = AutoProcessor.from_pretrained(self._config.model_path)

    def _build_qwen_messages(
        self,
        *,
        target_sample: ConstructionSiteSample,
        mode: str,
        example_samples: tuple[ConstructionSiteSample, ...],
        task_profile: str,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if mode == "five_shot":
            for example_sample in example_samples:
                prompt_text = _get_local_task_prompt(mode=mode, task_profile=task_profile)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": _load_pil_image(example_sample)},
                            {
                                "type": "text",
                                "text": f"{prompt_text}\nImage ID: {example_sample.image_id}",
                            },
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    build_author_style_example_answer(
                                        example_sample,
                                        task_profile=task_profile,
                                    ),
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                            }
                        ],
                    }
                )

        prompt_text = _get_local_task_prompt(mode=mode, task_profile=task_profile)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _load_pil_image(target_sample)},
                    {
                        "type": "text",
                        "text": f"{prompt_text}\nImage ID: {target_sample.image_id}",
                    },
                ],
            }
        )
        return messages


def run_local_qwen_baseline(
    *,
    client: LocalQwen3VLClient,
    model_name: str,
    target_samples: tuple[ConstructionSiteSample, ...],
    mode: str,
    example_samples: tuple[ConstructionSiteSample, ...],
    task_profile: str,
    show_progress: bool = False,
) -> list[Point1BaselineRecord]:
    """Run a local Qwen3-VL baseline over a sequence of target samples."""
    records: list[Point1BaselineRecord] = []
    total = len(target_samples)
    for index, target_sample in enumerate(target_samples, start=1):
        if show_progress:
            print(f"[{index}/{total}] running local {mode} on image {target_sample.image_id}")
        try:
            raw_response = client.complete_for_sample(
                target_sample=target_sample,
                mode=mode,
                example_samples=example_samples,
                task_profile=task_profile,
            )
            parsed_output = parse_prediction_set_response(raw_response, sample=target_sample)
            records.append(
                Point1BaselineRecord(
                    image_id=target_sample.image_id,
                    provider_name="local_qwen",
                    model_name=model_name,
                    mode=mode,
                    raw_response_text=raw_response,
                    parsed_output=parsed_output,
                )
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                Point1BaselineRecord(
                    image_id=target_sample.image_id,
                    provider_name="local_qwen",
                    model_name=model_name,
                    mode=mode,
                    raw_response_text=locals().get("raw_response", ""),
                    parsed_output=None,
                    error_message=str(exc),
                )
            )
    return records


def _load_pil_image(sample: ConstructionSiteSample):
    if sample.image is None or sample.image.bytes is None:
        raise ValueError(f"Sample {sample.image_id} does not contain embedded image bytes.")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - only triggered on missing local deps
        raise ImportError("Local Qwen3-VL inference requires Pillow.") from exc
    return Image.open(io.BytesIO(sample.image.bytes)).convert("RGB")


def _get_local_task_prompt(*, mode: str, task_profile: str) -> str:
    if mode == "five_shot":
        return get_five_shot_task_prompt(task_profile)
    return get_task_prompt(task_profile)
