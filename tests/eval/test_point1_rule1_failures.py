"""Tests for Rule 1 failure export helpers."""

from __future__ import annotations

from pathlib import Path

from benchmark.constructionsite10k.parser import parse_sample
from common.io.json_io import write_json
from eval.reports.point1_rule1_failures import export_rule1_failures


def test_export_rule1_failures_marks_fp_fn_and_unknown(
    sample_annotation: dict[str, object],
    tmp_path: Path,
) -> None:
    """Failure export should preserve binary FN status even when a sample is unknown."""
    sample_clean = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": b"clean-image", "path": "clean.jpg"},
            "rule_1_violation": None,
        }
    )
    sample_rule1 = parse_sample(
        {
            **sample_annotation,
            "image_id": "0000999",
            "image": {"bytes": b"rule1-image", "path": "rule1.jpg"},
            "rule_1_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "rule1"},
        }
    )
    output_path = tmp_path / "rule1.json"
    write_json(
        output_path,
        [
            {
                "image_id": sample_clean.image_id,
                "provider_name": "rule1_pipeline",
                "model_name": "demo",
                "mode": "rule1_smallloop",
                "raw_response_text": "",
                "parsed_output": {
                    "image_id": sample_clean.image_id,
                    "predictions": [
                        {
                            "rule_id": 1,
                            "decision_state": "violation",
                            "target_bbox": [0.1, 0.2, 0.3, 0.4],
                            "supporting_evidence_ids": [],
                            "counter_evidence_ids": [],
                            "unknown_items": [],
                            "reason_slots": {},
                            "reason_text": "fp",
                            "confidence": 0.9,
                        }
                    ],
                },
                "error_message": None,
            },
            {
                "image_id": sample_rule1.image_id,
                "provider_name": "rule1_pipeline",
                "model_name": "demo",
                "mode": "rule1_smallloop",
                "raw_response_text": "",
                "parsed_output": {
                    "image_id": sample_rule1.image_id,
                    "predictions": [
                        {
                            "rule_id": 1,
                            "decision_state": "unknown",
                            "target_bbox": None,
                            "supporting_evidence_ids": [],
                            "counter_evidence_ids": [],
                            "unknown_items": ["toe_covered"],
                            "reason_slots": {},
                            "reason_text": "unknown positive",
                            "confidence": 0.2,
                        }
                    ],
                },
                "error_message": None,
            },
        ],
    )

    export = export_rule1_failures(
        output_path=output_path,
        target_samples=(sample_clean, sample_rule1),
    )

    assert export["counts"] == {
        "false_positives": 1,
        "false_negatives": 1,
        "unknown_predictions": 1,
        "parse_failures": 0,
    }
    assert len(export["records"]) == 2
    positive_unknown = next(
        record for record in export["records"] if record["image_id"] == "0000999"
    )
    assert positive_unknown["is_false_negative"] is True
    assert positive_unknown["is_unknown_prediction"] is True
    assert positive_unknown["unknown_items"] == ["toe_covered"]
