"""Runner utilities for the Rule 1 small closed-loop pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.io.json_io import write_json
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
    progress_output: Path | None = None,
    checkpoint_output: Path | None = None,
    checkpoint_every: int | None = None,
    predicate_backend: str | None = None,
    candidate_batch_size: int | None = None,
    max_new_tokens: int | None = None,
    max_candidates_per_image: int | None = None,
) -> list[Point1BaselineRecord]:
    """Run the Rule 1 image-level pipeline over a sequence of benchmark samples."""
    active_pipeline = Rule1Pipeline() if pipeline is None else pipeline

    records: list[Point1BaselineRecord] = []
    total = len(target_samples)
    for index, target_sample in enumerate(target_samples, start=1):
        if show_progress:
            print(
                f"[{index}/{total}] running rule1_smallloop on image {target_sample.image_id}",
                flush=True,
            )
        started_at = perf_counter()
        try:
            result = active_pipeline.run(target_sample)
            total_ms = result.total_ms or (perf_counter() - started_at) * 1000.0
            records.append(
                Point1BaselineRecord(
                    image_id=target_sample.image_id,
                    provider_name=provider_name,
                    model_name=model_name,
                    mode=mode,
                    raw_response_text="",
                    parsed_output=result.to_prediction_set(),
                    candidate_ms=result.candidate_ms,
                    predicate_ms=result.predicate_ms,
                    executor_ms=result.executor_ms,
                    total_ms=total_ms,
                    candidate_count=result.candidate_count,
                    candidate_count_raw=result.candidate_count_raw,
                    candidate_count_capped=result.candidate_count_capped,
                    fallback_used=result.fallback_used,
                    predicate_backend=predicate_backend,
                    candidate_batch_size=candidate_batch_size,
                    max_new_tokens=max_new_tokens,
                    max_candidates_per_image=max_candidates_per_image,
                )
            )
        except Exception as exc:  # noqa: BLE001
            total_ms = (perf_counter() - started_at) * 1000.0
            records.append(
                Point1BaselineRecord(
                    image_id=target_sample.image_id,
                    provider_name=provider_name,
                    model_name=model_name,
                    mode=mode,
                    raw_response_text="",
                    parsed_output=None,
                    error_message=str(exc),
                    total_ms=total_ms,
                    predicate_backend=predicate_backend,
                    candidate_batch_size=candidate_batch_size,
                    max_new_tokens=max_new_tokens,
                    max_candidates_per_image=max_candidates_per_image,
                )
            )
        latest_record = records[-1]
        if progress_output is not None:
            write_json(
                progress_output,
                {
                    "total": total,
                    "completed": index,
                    "last_image_id": target_sample.image_id,
                    "candidate_ms": latest_record.candidate_ms,
                    "predicate_ms": latest_record.predicate_ms,
                    "executor_ms": latest_record.executor_ms,
                    "total_ms": latest_record.total_ms,
                    "candidate_count": latest_record.candidate_count,
                    "candidate_count_raw": latest_record.candidate_count_raw,
                    "candidate_count_capped": latest_record.candidate_count_capped,
                    "fallback_used": latest_record.fallback_used,
                    "predicate_backend": latest_record.predicate_backend,
                    "candidate_batch_size": latest_record.candidate_batch_size,
                    "max_new_tokens": latest_record.max_new_tokens,
                    "max_candidates_per_image": latest_record.max_candidates_per_image,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
        if (
            checkpoint_output is not None
            and checkpoint_every is not None
            and checkpoint_every > 0
            and (index % checkpoint_every == 0 or index == total)
        ):
            write_json(checkpoint_output, [record.to_dict() for record in records])
    return records
