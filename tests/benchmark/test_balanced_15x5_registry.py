"""Tests for the frozen balanced_15x5 registry file."""

from __future__ import annotations

from pathlib import Path

from benchmark.constructionsite10k.registry import SplitRegistry


def test_balanced_15x5_registry_has_expected_bucket_sizes() -> None:
    """The frozen quick-test subset should contain five disjoint buckets of 15 IDs each."""
    registry = SplitRegistry.from_json(
        Path("src/benchmark/splits/constructionsite10k_balanced_15x5.json")
    )

    bucket_names = (
        "balanced_15x5_clean",
        "balanced_15x5_rule1",
        "balanced_15x5_rule2",
        "balanced_15x5_rule3",
        "balanced_15x5_rule4",
    )
    all_ids: list[str] = []
    for bucket_name in bucket_names:
        bucket = registry.get_split(bucket_name)
        assert len(bucket) == 15
        all_ids.extend(bucket)

    assert len(registry.get_split("balanced_15x5")) == 75
    assert len(set(all_ids)) == 75
