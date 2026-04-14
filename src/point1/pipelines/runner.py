"""Runner utilities for the Rule 1 small closed-loop pipeline."""

from __future__ import annotations

from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.schemas.point1 import Point1BaselineRecord
from point1.pipelines.rule1 import Rule1Pipeline


def run_rule1_pipeline(
    *,
    target_samples: tuple[ConstructionSiteSample, ...],
    pipeline: Rule1Pipeline | object | None = None,
    provider_name: str = "rule1_pipeline",
    model_name: str = "opencv_hog+heuristic_rule1",
    mode: str = "rule1_smallloop",
    show_progress: bool = False,
) -> list[Point1BaselineRecord]:
    """Run the Rule 1 image-level pipeline over a sequence of benchmark samples."""
    active_pipeline = Rule1Pipeline() if pipeline is None else pipeline

    records: list[Point1BaselineRecord] = []
    total = len(target_samples)
    for index, target_sample in enumerate(target_samples, start=1):
        if show_progress:
            print(f"[{index}/{total}] running rule1_smallloop on image {target_sample.image_id}")
        try:
            result = active_pipeline.run(target_sample)
            records.append(
                Point1BaselineRecord(
                    image_id=target_sample.image_id,
                    provider_name=provider_name,
                    model_name=model_name,
                    mode=mode,
                    raw_response_text="",
                    parsed_output=result.to_prediction_set(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                Point1BaselineRecord(
                    image_id=target_sample.image_id,
                    provider_name=provider_name,
                    model_name=model_name,
                    mode=mode,
                    raw_response_text="",
                    parsed_output=None,
                    error_message=str(exc),
                )
            )
    return records
