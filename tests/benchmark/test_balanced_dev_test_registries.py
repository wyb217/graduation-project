"""Tests for the frozen balanced dev/test registries."""

from __future__ import annotations

from pathlib import Path

from benchmark.constructionsite10k.registry import SplitRegistry


def test_balanced_dev_and_test_registries_have_expected_sizes() -> None:
    """Frozen quick-test registries should expose stable disjoint bucket sizes."""
    expected_configs = (
        ("constructionsite10k_balanced_dev_15x5.json", 15, 75),
        ("constructionsite10k_balanced_test_13x5.json", 13, 65),
    )
    for registry_name, bucket_size, total_size in expected_configs:
        registry = SplitRegistry.from_json(Path("src/benchmark/splits") / registry_name)
        subset_prefix = registry_name.removeprefix("constructionsite10k_").removesuffix(".json")
        bucket_names = (
            f"{subset_prefix}_clean",
            f"{subset_prefix}_rule1",
            f"{subset_prefix}_rule2",
            f"{subset_prefix}_rule3",
            f"{subset_prefix}_rule4",
        )
        all_ids: list[str] = []
        for bucket_name in bucket_names:
            bucket = registry.get_split(bucket_name)
            assert len(bucket) == bucket_size
            all_ids.extend(bucket)

        assert len(registry.get_split(subset_prefix)) == total_size
        assert len(set(all_ids)) == total_size
