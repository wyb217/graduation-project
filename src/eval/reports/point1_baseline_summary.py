"""Utilities for summarizing Point 1 API baseline results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmark.constructionsite10k.registry import SplitRegistry

BUCKET_NAMES = ("clean", "rule1", "rule2", "rule3", "rule4")


@dataclass(frozen=True, slots=True)
class RuleMetrics:
    """Precision/recall/F1 metrics for one rule."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def summarize_baseline_run(
    *,
    output_path: Path,
    registry_path: Path,
    subset_name: str,
) -> dict[str, object]:
    """Compute parse success, bucket hit rates, and rule-level classification metrics."""
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    registry = SplitRegistry.from_json(registry_path)
    truth = _build_truth_map(registry=registry, subset_name=subset_name)

    records = list(payload)
    parsed_records = [record for record in records if record["parsed_output"] is not None]
    parse_success_rate = _safe_divide(len(parsed_records), len(records))

    bucket_hits = _compute_bucket_hits(records=records, truth=truth)
    rule_metrics = _compute_rule_metrics(records=records, truth=truth)
    macro_f1 = _safe_divide(sum(metric.f1 for metric in rule_metrics.values()), len(rule_metrics))

    return {
        "num_records": len(records),
        "num_parsed": len(parsed_records),
        "parse_success_rate": parse_success_rate,
        "bucket_hit_rate": bucket_hits,
        "rule_metrics": {
            str(rule_id): {
                "precision": metric.precision,
                "recall": metric.recall,
                "f1": metric.f1,
                "tp": metric.tp,
                "fp": metric.fp,
                "fn": metric.fn,
            }
            for rule_id, metric in rule_metrics.items()
        },
        "macro_f1": macro_f1,
    }


def _build_truth_map(*, registry: SplitRegistry, subset_name: str) -> dict[str, str]:
    truth: dict[str, str] = {}
    for image_id in registry.get_split(f"{subset_name}_clean"):
        truth[image_id] = "clean"
    for rule_id in range(1, 5):
        for image_id in registry.get_split(f"{subset_name}_rule{rule_id}"):
            truth[image_id] = f"rule{rule_id}"
    return truth


def _compute_bucket_hits(
    *,
    records: list[dict[str, object]],
    truth: dict[str, str],
) -> dict[str, dict[str, float | int]]:
    bucket_correct = {bucket: 0 for bucket in BUCKET_NAMES}
    bucket_total = {bucket: 0 for bucket in BUCKET_NAMES}

    for record in records:
        image_id = str(record["image_id"])
        bucket_name = truth[image_id]
        bucket_total[bucket_name] += 1
        parsed_output = record["parsed_output"]
        if parsed_output is None:
            continue
        predictions = {
            int(prediction["rule_id"]): str(prediction["decision_state"])
            for prediction in parsed_output["predictions"]
        }
        if bucket_name == "clean":
            is_correct = all(state != "violation" for state in predictions.values())
        else:
            active_rule_id = int(bucket_name[-1])
            is_correct = predictions.get(active_rule_id) == "violation"
        bucket_correct[bucket_name] += int(is_correct)

    return {
        bucket_name: {
            "correct": bucket_correct[bucket_name],
            "total": bucket_total[bucket_name],
            "hit_rate": _safe_divide(bucket_correct[bucket_name], bucket_total[bucket_name]),
        }
        for bucket_name in BUCKET_NAMES
    }


def _compute_rule_metrics(
    *,
    records: list[dict[str, object]],
    truth: dict[str, str],
) -> dict[int, RuleMetrics]:
    counters = {rule_id: {"tp": 0, "fp": 0, "fn": 0} for rule_id in range(1, 5)}

    for record in records:
        image_id = str(record["image_id"])
        truth_bucket = truth[image_id]
        truth_rule = None if truth_bucket == "clean" else int(truth_bucket[-1])
        parsed_output = record["parsed_output"]
        predictions = {}
        if parsed_output is not None:
            predictions = {
                int(prediction["rule_id"]): str(prediction["decision_state"])
                for prediction in parsed_output["predictions"]
            }

        for rule_id in range(1, 5):
            predicted_positive = predictions.get(rule_id) == "violation"
            actual_positive = truth_rule == rule_id
            if predicted_positive and actual_positive:
                counters[rule_id]["tp"] += 1
            elif predicted_positive and not actual_positive:
                counters[rule_id]["fp"] += 1
            elif (not predicted_positive) and actual_positive:
                counters[rule_id]["fn"] += 1

    metrics: dict[int, RuleMetrics] = {}
    for rule_id, counts in counters.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        metrics[rule_id] = RuleMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp,
            fp=fp,
            fn=fn,
        )
    return metrics


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
