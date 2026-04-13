"""Utilities for summarizing Point 1 baseline results."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from benchmark.constructionsite10k.loader import ConstructionSite10kDataset
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
    """Compute parse success, bucket hit rates, and rule-level metrics on a frozen subset."""
    payload = _load_records(output_path)
    registry = SplitRegistry.from_json(registry_path)
    truth = _build_single_label_truth_map(registry=registry, subset_name=subset_name)

    summary = _build_multilabel_summary(records=payload, truth=truth, truth_source="registry")
    summary["bucket_hit_rate"] = _compute_bucket_hits(records=payload, truth=truth)
    return summary


def summarize_baseline_run_from_dataset(
    *,
    output_path: Path,
    target_parquet_paths: Iterable[Path],
    registry_path: Path | None = None,
    split_name: str | None = None,
) -> dict[str, object]:
    """Compute rule metrics against dataset-backed truth, including multi-label cases."""
    registry = SplitRegistry.from_json(registry_path) if registry_path is not None else None
    dataset = ConstructionSite10kDataset.from_parquet(
        target_parquet_paths,
        registry=registry,
        split_name=split_name,
        include_image_bytes=False,
    )
    truth = {
        sample.image_id: frozenset(
            rule_id for rule_id, violation in sample.violations.items() if violation is not None
        )
        for sample in dataset.samples
    }
    summary = _build_multilabel_summary(
        records=_load_records(output_path),
        truth=truth,
        truth_source="dataset",
    )
    summary["multi_label_cardinality"] = _build_multi_label_cardinality(truth)
    summary["per_rule_positive_support"] = {
        str(rule_id): sum(rule_id in active_rules for active_rules in truth.values())
        for rule_id in range(1, 5)
    }
    if split_name is not None:
        summary["target_split"] = split_name
    return summary


def _build_multilabel_summary(
    *,
    records: list[dict[str, object]],
    truth: dict[str, frozenset[int]],
    truth_source: str,
) -> dict[str, object]:
    parsed_records = [record for record in records if record["parsed_output"] is not None]
    parse_success_rate = _safe_divide(len(parsed_records), len(records))
    rule_metrics = _compute_rule_metrics(records=records, truth=truth)
    macro_f1 = _safe_divide(sum(metric.f1 for metric in rule_metrics.values()), len(rule_metrics))

    return {
        "truth_source": truth_source,
        "num_records": len(records),
        "num_parsed": len(parsed_records),
        "parse_success_rate": parse_success_rate,
        "rule_metrics": _serialize_rule_metrics(rule_metrics),
        "rule_table_markdown": format_rule_metrics_markdown(rule_metrics),
        "macro_f1": macro_f1,
    }


def format_rule_metrics_markdown(rule_metrics: dict[int, RuleMetrics]) -> str:
    """Render per-rule metrics as a paper-friendly Markdown table."""
    header = [
        "| Rule | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [
        (
            f"| rule{rule_id} | {metric.precision:.3f} | {metric.recall:.3f} | "
            f"{metric.f1:.3f} | {metric.tp} | {metric.fp} | {metric.fn} |"
        )
        for rule_id, metric in sorted(rule_metrics.items())
    ]
    return "\n".join([*header, *rows])


def _load_records(output_path: Path) -> list[dict[str, object]]:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return list(payload)


def _build_single_label_truth_map(
    *,
    registry: SplitRegistry,
    subset_name: str,
) -> dict[str, frozenset[int]]:
    truth: dict[str, frozenset[int]] = {}
    for image_id in registry.get_split(f"{subset_name}_clean"):
        truth[image_id] = frozenset()
    for rule_id in range(1, 5):
        for image_id in registry.get_split(f"{subset_name}_rule{rule_id}"):
            truth[image_id] = frozenset({rule_id})
    return truth


def _compute_bucket_hits(
    *,
    records: list[dict[str, object]],
    truth: dict[str, frozenset[int]],
) -> dict[str, dict[str, float | int]]:
    bucket_correct = {bucket: 0 for bucket in BUCKET_NAMES}
    bucket_total = {bucket: 0 for bucket in BUCKET_NAMES}

    for record in records:
        image_id = str(record["image_id"])
        truth_rules = truth[image_id]
        bucket_name = "clean" if not truth_rules else f"rule{next(iter(sorted(truth_rules)))}"
        bucket_total[bucket_name] += 1
        parsed_output = record["parsed_output"]
        if parsed_output is None:
            continue
        predicted_rules = _extract_predicted_positive_rules(parsed_output)
        is_correct = predicted_rules == truth_rules
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
    truth: dict[str, frozenset[int]],
) -> dict[int, RuleMetrics]:
    counters = {rule_id: {"tp": 0, "fp": 0, "fn": 0} for rule_id in range(1, 5)}

    for record in records:
        image_id = str(record["image_id"])
        truth_rules = truth[image_id]
        parsed_output = record["parsed_output"]
        predicted_rules = frozenset()
        if parsed_output is not None:
            predicted_rules = _extract_predicted_positive_rules(parsed_output)

        for rule_id in range(1, 5):
            predicted_positive = rule_id in predicted_rules
            actual_positive = rule_id in truth_rules
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


def _extract_predicted_positive_rules(parsed_output: dict[str, object]) -> frozenset[int]:
    return frozenset(
        int(prediction["rule_id"])
        for prediction in parsed_output["predictions"]
        if str(prediction["decision_state"]) == "violation"
    )


def _build_multi_label_cardinality(
    truth: dict[str, frozenset[int]],
) -> dict[str, int]:
    counts = Counter(len(active_rules) for active_rules in truth.values())
    return {str(cardinality): counts.get(cardinality, 0) for cardinality in range(5)}


def _serialize_rule_metrics(
    rule_metrics: dict[int, RuleMetrics],
) -> dict[str, dict[str, float | int]]:
    return {
        str(rule_id): {
            "precision": metric.precision,
            "recall": metric.recall,
            "f1": metric.f1,
            "tp": metric.tp,
            "fp": metric.fp,
            "fn": metric.fn,
        }
        for rule_id, metric in rule_metrics.items()
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
