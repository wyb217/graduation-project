"""Failure export helpers for Rule 1 runs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmark.constructionsite10k.types import ConstructionSiteSample


def export_rule1_failures(
    *,
    output_path: Path,
    target_samples: Iterable[ConstructionSiteSample],
) -> dict[str, object]:
    """Export FP/FN/unknown/parse-fail samples for downstream Rule 1 analysis."""
    records = json.loads(output_path.read_text(encoding="utf-8"))
    truth = {sample.image_id: bool(sample.violations[1] is not None) for sample in target_samples}

    counts = {
        "false_positives": 0,
        "false_negatives": 0,
        "unknown_predictions": 0,
        "parse_failures": 0,
    }
    failure_records: list[dict[str, object]] = []

    for record in records:
        image_id = str(record["image_id"])
        actual_positive = truth[image_id]
        parsed_output = record.get("parsed_output")
        selected_prediction = _extract_selected_rule1_prediction(parsed_output)
        decision_state = (
            "parse_fail"
            if parsed_output is None
            else (
                "no_violation"
                if selected_prediction is None
                else selected_prediction["decision_state"]
            )
        )
        predicted_positive = decision_state == "violation"
        is_parse_fail = parsed_output is None
        is_unknown_prediction = decision_state == "unknown"
        is_false_positive = predicted_positive and not actual_positive
        is_false_negative = (not predicted_positive) and actual_positive

        counts["false_positives"] += int(is_false_positive)
        counts["false_negatives"] += int(is_false_negative)
        counts["unknown_predictions"] += int(is_unknown_prediction)
        counts["parse_failures"] += int(is_parse_fail)

        if not (is_false_positive or is_false_negative or is_unknown_prediction or is_parse_fail):
            continue

        failure_records.append(
            {
                "image_id": image_id,
                "actual_rule1_positive": actual_positive,
                "decision_state": decision_state,
                "is_false_positive": is_false_positive,
                "is_false_negative": is_false_negative,
                "is_unknown_prediction": is_unknown_prediction,
                "is_parse_fail": is_parse_fail,
                "unknown_items": (
                    [] if selected_prediction is None else selected_prediction["unknown_items"]
                ),
                "reason_text": (
                    "" if selected_prediction is None else selected_prediction["reason_text"]
                ),
                "target_bbox": (
                    None if selected_prediction is None else selected_prediction["target_bbox"]
                ),
                "confidence": (
                    None if selected_prediction is None else selected_prediction["confidence"]
                ),
                "error_message": record.get("error_message"),
            }
        )

    return {
        "counts": counts,
        "records": failure_records,
    }


def _extract_selected_rule1_prediction(parsed_output: object) -> dict[str, Any] | None:
    if not isinstance(parsed_output, dict):
        return None
    predictions = parsed_output.get("predictions")
    if not isinstance(predictions, list):
        return None

    selected_prediction: dict[str, Any] | None = None
    for raw_prediction in predictions:
        if not isinstance(raw_prediction, dict) or int(raw_prediction.get("rule_id", -1)) != 1:
            continue
        if selected_prediction is None or (
            _decision_priority(str(raw_prediction.get("decision_state", "")))
            > _decision_priority(str(selected_prediction.get("decision_state", "")))
        ):
            selected_prediction = raw_prediction
    return selected_prediction


def _decision_priority(decision_state: str) -> int:
    priority = {
        "no_violation": 0,
        "unknown": 1,
        "violation": 2,
    }
    return priority.get(decision_state, -1)
