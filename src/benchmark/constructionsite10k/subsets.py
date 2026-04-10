"""Helpers for building frozen benchmark subsets."""

from collections.abc import Iterable

from benchmark.constructionsite10k.types import ConstructionSiteSample

BUCKET_ORDER = ("clean", "rule1", "rule2", "rule3", "rule4")


def build_balanced_subset_registry(
    samples: Iterable[ConstructionSiteSample],
    *,
    per_bucket: int = 15,
    subset_name: str = "balanced_15x5",
) -> dict[str, tuple[str, ...]]:
    """Build a frozen balanced subset from clean and single-rule samples."""
    buckets: dict[str, list[str]] = {bucket_name: [] for bucket_name in BUCKET_ORDER}

    for sample in samples:
        active_rules = sorted(
            rule_id for rule_id, violation in sample.violations.items() if violation is not None
        )
        if not active_rules:
            buckets["clean"].append(sample.image_id)
            continue
        if len(active_rules) == 1:
            buckets[f"rule{active_rules[0]}"].append(sample.image_id)

    registry: dict[str, tuple[str, ...]] = {}
    all_ids: list[str] = []
    for bucket_name in BUCKET_ORDER:
        selected_ids = tuple(sorted(buckets[bucket_name])[:per_bucket])
        if len(selected_ids) < per_bucket:
            raise ValueError(
                "Not enough samples for "
                f"{bucket_name}: expected {per_bucket}, got {len(selected_ids)}."
            )
        registry[f"{subset_name}_{bucket_name}"] = selected_ids
        all_ids.extend(selected_ids)

    registry[subset_name] = tuple(all_ids)
    return registry
