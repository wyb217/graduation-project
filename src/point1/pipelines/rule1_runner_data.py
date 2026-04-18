"""Target loading and summary helpers for the Rule 1 runner script."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmark.constructionsite10k.loader import ConstructionSite10kDataset
from benchmark.constructionsite10k.registry import SplitRegistry
from benchmark.constructionsite10k.types import ConstructionSiteSample
from eval.reports.point1_rule1_summary import (
    summarize_rule1_bucketed_run,
    summarize_rule1_run_from_dataset,
    summarize_rule1_smallloop,
)


def load_target_samples(
    args: argparse.Namespace,
) -> tuple[tuple[ConstructionSiteSample, ...], dict[str, object]]:
    """Load the Rule 1 target samples from fulltest or registry-driven subsets."""
    if args.fulltest:
        if args.target_split_names:
            raise ValueError("--target-split-names cannot be used with --fulltest.")
        if args.registry is not None:
            raise ValueError("--registry cannot be used with --fulltest.")
        dataset = ConstructionSite10kDataset.from_parquet(
            args.target_parquet,
            include_image_bytes=True,
        )
        target_samples = dataset.samples if args.limit is None else dataset.samples[: args.limit]
        return tuple(target_samples), {"mode": "fulltest"}

    if args.target_split_names is None or len(args.target_split_names) < 2:
        raise ValueError("Rule 1 run expects at least two split names unless --fulltest is set.")
    if args.registry is None:
        raise ValueError("--registry is required unless --fulltest is set.")

    registry = SplitRegistry.from_json(args.registry)
    positive_split_name = resolve_positive_split_name(args)
    target_image_ids = merge_split_image_ids(registry, tuple(args.target_split_names))
    dataset = ConstructionSite10kDataset.from_parquet(
        args.target_parquet,
        include_image_bytes=True,
        image_ids=target_image_ids,
    )
    available_sample_ids = {sample.image_id for sample in dataset.samples}
    ordered_samples = tuple(
        dataset.get_by_image_id(image_id)
        for image_id in target_image_ids
        if image_id in available_sample_ids
    )
    target_samples = ordered_samples if args.limit is None else ordered_samples[: args.limit]
    return target_samples, {
        "mode": "subset",
        "registry": registry,
        "positive_split_name": positive_split_name,
    }


def build_rule1_summary(
    *,
    args: argparse.Namespace,
    output_path: Path,
    target_samples: tuple[ConstructionSiteSample, ...],
    summary_context: dict[str, object],
) -> dict[str, object]:
    """Build the Rule 1 summary for either fulltest or bucketed subset runs."""
    if summary_context["mode"] == "fulltest":
        summary = summarize_rule1_run_from_dataset(
            output_path=output_path,
            target_parquet_paths=tuple(args.target_parquet),
        )
        return _attach_runtime_summary(output_path=output_path, summary=summary)

    positive_split_name = str(summary_context["positive_split_name"])
    if (
        args.limit is None
        and len(args.target_split_names) == 2
        and positive_split_name == args.target_split_names[1]
    ):
        return summarize_rule1_smallloop(
            output_path=output_path,
            registry_path=args.registry,
            clean_split_name=args.target_split_names[0],
            positive_split_name=positive_split_name,
        )

    registry = summary_context["registry"]
    selected_image_ids = tuple(sample.image_id for sample in target_samples)
    selected_image_id_set = set(selected_image_ids)
    bucket_image_ids = {
        split_name: tuple(
            image_id
            for image_id in registry.get_split(split_name)
            if image_id in selected_image_id_set
        )
        for split_name in args.target_split_names
    }
    summary = summarize_rule1_bucketed_run(
        output_path=output_path,
        bucket_image_ids=bucket_image_ids,
        positive_bucket_name=positive_split_name,
    )
    return _attach_runtime_summary(output_path=output_path, summary=summary)


def _attach_runtime_summary(
    *,
    output_path: Path,
    summary: dict[str, object],
) -> dict[str, object]:
    records = json.loads(output_path.read_text(encoding="utf-8"))
    candidate_values = _collect_numeric(records, "candidate_ms")
    predicate_values = _collect_numeric(records, "predicate_ms")
    executor_values = _collect_numeric(records, "executor_ms")
    total_values = _collect_numeric(records, "total_ms")
    candidate_count_values = _collect_numeric(records, "candidate_count")

    fallback_values = [
        bool(record["fallback_used"])
        for record in records
        if record.get("fallback_used") is not None
    ]
    predicate_backend = _first_present(records, "predicate_backend")
    candidate_batch_size = _first_present(records, "candidate_batch_size")
    max_new_tokens = _first_present(records, "max_new_tokens")

    return {
        **summary,
        "timing_ms": {
            "candidate": _build_distribution(candidate_values),
            "predicate": _build_distribution(predicate_values),
            "executor": _build_distribution(executor_values),
            "total": _build_distribution(total_values),
        },
        "candidate_count_stats": _build_distribution(candidate_count_values),
        "fallback_rate": (
            sum(fallback_values) / len(fallback_values) if fallback_values else None
        ),
        "runtime_config": {
            "predicate_backend": predicate_backend,
            "candidate_batch_size": candidate_batch_size,
            "max_new_tokens": max_new_tokens,
        },
    }


def _collect_numeric(records: list[dict[str, object]], field_name: str) -> list[float]:
    return [
        float(record[field_name])
        for record in records
        if record.get(field_name) is not None
    ]


def _first_present(records: list[dict[str, object]], field_name: str):
    for record in records:
        value = record.get(field_name)
        if value is not None:
            return value
    return None


def _build_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": _nearest_rank_percentile(ordered, 0.50),
        "p95": _nearest_rank_percentile(ordered, 0.95),
    }


def _nearest_rank_percentile(values: list[float], quantile: float) -> float:
    rank = max(1, math.ceil(len(values) * quantile))
    return values[rank - 1]


def merge_split_image_ids(
    registry: SplitRegistry,
    split_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Merge multiple split names into one ordered, deduplicated image-id tuple."""
    seen: set[str] = set()
    ordered_ids: list[str] = []
    for split_name in split_names:
        for image_id in registry.get_split(split_name):
            if image_id in seen:
                continue
            seen.add(image_id)
            ordered_ids.append(image_id)
    return tuple(ordered_ids)


def resolve_positive_split_name(args: argparse.Namespace) -> str:
    """Resolve the positive Rule 1 split name from explicit args or defaults."""
    if args.positive_split_name is not None:
        if args.positive_split_name not in args.target_split_names:
            raise ValueError("--positive-split-name must be one of --target-split-names.")
        return str(args.positive_split_name)
    if len(args.target_split_names) == 2:
        return str(args.target_split_names[1])
    raise ValueError("Multi-bucket runs require --positive-split-name.")
