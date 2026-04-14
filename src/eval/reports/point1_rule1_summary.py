"""Summary helpers for Rule 1 closed-loop runs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from benchmark.constructionsite10k.loader import ConstructionSite10kDataset
from benchmark.constructionsite10k.registry import SplitRegistry


def summarize_rule1_smallloop(
    *,
    output_path: Path,
    registry_path: Path | None = None,
    clean_split_name: str | None = None,
    positive_split_name: str | None = None,
    clean_image_ids: tuple[str, ...] | None = None,
    positive_image_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Summarize a two-bucket Rule 1 run on clean vs rule1 samples."""
    if clean_image_ids is not None or positive_image_ids is not None:
        if clean_image_ids is None or positive_image_ids is None:
            raise ValueError("clean_image_ids and positive_image_ids must be provided together.")
        bucket_image_ids = {
            str(clean_split_name or "clean"): clean_image_ids,
            str(positive_split_name or "rule1"): positive_image_ids,
        }
        resolved_positive_name = str(positive_split_name or "rule1")
        resolved_clean_name = str(clean_split_name or "clean")
    else:
        if registry_path is None or clean_split_name is None or positive_split_name is None:
            raise ValueError(
                "Rule 1 summary requires either explicit image IDs or registry_path + split names."
            )
        registry = SplitRegistry.from_json(registry_path)
        bucket_image_ids = {
            clean_split_name: tuple(registry.get_split(clean_split_name)),
            positive_split_name: tuple(registry.get_split(positive_split_name)),
        }
        resolved_positive_name = positive_split_name
        resolved_clean_name = clean_split_name

    summary = summarize_rule1_bucketed_run(
        output_path=output_path,
        bucket_image_ids=bucket_image_ids,
        positive_bucket_name=resolved_positive_name,
    )
    return {
        **summary,
        "clean_hit_rate": summary["bucket_hit_rate"][resolved_clean_name]["hit_rate"],
        "rule1_hit_rate": summary["bucket_hit_rate"][resolved_positive_name]["hit_rate"],
        "unknown_rate_clean": summary["unknown_rate_by_bucket"][resolved_clean_name],
        "unknown_rate_rule1": summary["unknown_rate_by_bucket"][resolved_positive_name],
        "clean_split_name": resolved_clean_name,
        "positive_split_name": resolved_positive_name,
    }


def summarize_rule1_bucketed_run(
    *,
    output_path: Path,
    bucket_image_ids: dict[str, tuple[str, ...]],
    positive_bucket_name: str,
) -> dict[str, object]:
    """Summarize a Rule 1 run across arbitrary frozen buckets."""
    if positive_bucket_name not in bucket_image_ids:
        raise ValueError(f"Unknown positive bucket: {positive_bucket_name}")

    truth = {
        image_id: bucket_name
        for bucket_name, image_ids in bucket_image_ids.items()
        for image_id in image_ids
    }
    records = json.loads(output_path.read_text(encoding="utf-8"))

    tp = fp = fn = 0
    num_parsed = 0
    state_counts = {"violation": 0, "unknown": 0, "no_violation": 0, "parse_fail": 0}
    bucket_correct = {bucket_name: 0 for bucket_name in bucket_image_ids}
    bucket_total = {
        bucket_name: len(image_ids) for bucket_name, image_ids in bucket_image_ids.items()
    }
    bucket_unknown = {bucket_name: 0 for bucket_name in bucket_image_ids}
    fp_by_bucket = {
        bucket_name: 0 for bucket_name in bucket_image_ids if bucket_name != positive_bucket_name
    }

    for record in records:
        image_id = str(record["image_id"])
        bucket_name = truth[image_id]
        parsed_output = record.get("parsed_output")

        if parsed_output is None:
            state_counts["parse_fail"] += 1
            decision_state = "no_violation"
        else:
            num_parsed += 1
            decision_state = _extract_rule1_decision_state(parsed_output)
            state_counts[decision_state] = state_counts.get(decision_state, 0) + 1

        predicted_positive = decision_state == "violation"
        actual_positive = bucket_name == positive_bucket_name

        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
            fp_by_bucket[bucket_name] += 1
        elif (not predicted_positive) and actual_positive:
            fn += 1

        if parsed_output is not None:
            expected_state = "violation" if actual_positive else "no_violation"
            bucket_correct[bucket_name] += int(decision_state == expected_state)
            bucket_unknown[bucket_name] += int(decision_state == "unknown")

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    negative_bucket_names = [
        bucket_name for bucket_name in bucket_image_ids if bucket_name != positive_bucket_name
    ]
    negative_total = sum(bucket_total[bucket_name] for bucket_name in negative_bucket_names)
    negative_correct = sum(bucket_correct[bucket_name] for bucket_name in negative_bucket_names)

    return {
        "rule_id": 1,
        "num_records": len(records),
        "num_parsed": num_parsed,
        "parse_success_rate": _safe_divide(num_parsed, len(records)),
        "rule1_precision": precision,
        "rule1_recall": recall,
        "rule1_f1": f1,
        "rule1_tp": tp,
        "rule1_fp": fp,
        "rule1_fn": fn,
        "negative_hit_rate": _safe_divide(negative_correct, negative_total),
        "state_counts": state_counts,
        "bucket_hit_rate": {
            bucket_name: {
                "correct": bucket_correct[bucket_name],
                "total": bucket_total[bucket_name],
                "hit_rate": _safe_divide(bucket_correct[bucket_name], bucket_total[bucket_name]),
            }
            for bucket_name in bucket_image_ids
        },
        "unknown_rate_overall": _safe_divide(
            sum(bucket_unknown.values()),
            len(records),
        ),
        "unknown_rate_by_bucket": {
            bucket_name: _safe_divide(bucket_unknown[bucket_name], bucket_total[bucket_name])
            for bucket_name in bucket_image_ids
        },
        "fp_by_bucket": fp_by_bucket,
        "positive_bucket_name": positive_bucket_name,
    }


def summarize_rule1_run_from_dataset(
    *,
    output_path: Path,
    target_parquet_paths: Iterable[Path],
) -> dict[str, object]:
    """Summarize a Rule 1 run directly against dataset-backed truth."""
    dataset = ConstructionSite10kDataset.from_parquet(
        target_parquet_paths,
        include_image_bytes=False,
    )
    truth = {
        sample.image_id: bool(sample.violations[1] is not None)
        for sample in dataset.samples
    }
    records = json.loads(output_path.read_text(encoding="utf-8"))

    tp = fp = fn = 0
    num_parsed = 0
    state_counts = {"violation": 0, "unknown": 0, "no_violation": 0, "parse_fail": 0}
    positive_support = sum(truth.values())
    negative_support = len(truth) - positive_support

    for record in records:
        image_id = str(record["image_id"])
        actual_positive = truth[image_id]
        parsed_output = record.get("parsed_output")

        if parsed_output is None:
            decision_state = "no_violation"
            state_counts["parse_fail"] += 1
        else:
            num_parsed += 1
            decision_state = _extract_rule1_decision_state(parsed_output)
            state_counts[decision_state] = state_counts.get(decision_state, 0) + 1

        predicted_positive = decision_state == "violation"
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif (not predicted_positive) and actual_positive:
            fn += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "rule_id": 1,
        "truth_source": "dataset",
        "num_records": len(records),
        "num_parsed": num_parsed,
        "parse_success_rate": _safe_divide(num_parsed, len(records)),
        "rule1_precision": precision,
        "rule1_recall": recall,
        "rule1_f1": f1,
        "rule1_tp": tp,
        "rule1_fp": fp,
        "rule1_fn": fn,
        "state_counts": state_counts,
        "unknown_rate_overall": _safe_divide(state_counts["unknown"], len(records)),
        "positive_support": positive_support,
        "negative_support": negative_support,
    }


def _extract_rule1_decision_state(parsed_output: object) -> str:
    if not isinstance(parsed_output, dict):
        return "no_violation"
    predictions = parsed_output.get("predictions", [])
    if not isinstance(predictions, list):
        return "no_violation"

    best_state: str | None = None
    for raw_prediction in predictions:
        if not isinstance(raw_prediction, dict) or int(raw_prediction.get("rule_id", -1)) != 1:
            continue
        decision_state = str(raw_prediction.get("decision_state", ""))
        if best_state is None or (
            _decision_priority(decision_state) > _decision_priority(best_state)
        ):
            best_state = decision_state
    return "no_violation" if best_state is None else best_state


def _decision_priority(decision_state: str) -> int:
    priority = {
        "no_violation": 0,
        "unknown": 1,
        "violation": 2,
    }
    return priority.get(decision_state, -1)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
