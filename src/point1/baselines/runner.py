"""Runner utilities for Point 1 API baselines."""

from __future__ import annotations

from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.schemas.point1 import Point1BaselineRecord
from point1.baselines.client import VisionLanguageClient
from point1.baselines.parsing import parse_prediction_set_response
from point1.baselines.prompting import build_inference_messages


def run_api_baseline(
    *,
    client: VisionLanguageClient,
    model_name: str,
    provider_name: str,
    target_samples: tuple[ConstructionSiteSample, ...],
    mode: str,
    example_samples: tuple[ConstructionSiteSample, ...],
    temperature: float = 0.0,
    max_tokens: int = 1200,
    show_progress: bool = False,
    task_profile: str = "structured",
) -> list[Point1BaselineRecord]:
    """Run one API baseline over a sequence of target samples."""
    records: list[Point1BaselineRecord] = []
    total = len(target_samples)
    for index, target_sample in enumerate(target_samples, start=1):
        if show_progress:
            print(f"[{index}/{total}] running {mode} on image {target_sample.image_id}")
        try:
            messages = build_inference_messages(
                target_sample=target_sample,
                mode=mode,
                example_samples=example_samples,
                task_profile=task_profile,
            )
            raw_response = client.complete(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed_output = parse_prediction_set_response(raw_response, sample=target_sample)
            records.append(
                Point1BaselineRecord(
                    image_id=target_sample.image_id,
                    provider_name=provider_name,
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
                    provider_name=provider_name,
                    model_name=model_name,
                    mode=mode,
                    raw_response_text=locals().get("raw_response", ""),
                    parsed_output=None,
                    error_message=str(exc),
                )
            )
    return records
