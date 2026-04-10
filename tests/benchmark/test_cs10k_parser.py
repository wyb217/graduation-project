"""Tests for ConstructionSite10k annotation parsing."""

from __future__ import annotations

from typing import Any

from benchmark.constructionsite10k.parser import parse_sample


def test_parse_sample_reads_rule_and_object_boxes(
    sample_annotation: dict[str, Any],
) -> None:
    """The parser should preserve rule and object box semantics."""
    sample = parse_sample(sample_annotation)

    assert sample.image_id == "0000424"
    assert sample.violations[1] is not None
    assert sample.violations[1].reason.startswith("The worker")
    assert sample.violations[2] is None
    assert "excavator" in sample.object_boxes
    assert sample.object_boxes["worker_with_white_hard_hat"][0].to_list() == [
        0.22,
        0.59,
        0.28,
        0.75,
    ]
