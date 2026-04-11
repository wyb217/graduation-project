"""Tests for balanced subset selection helpers."""

from __future__ import annotations

from benchmark.constructionsite10k.subsets import build_balanced_subset_registry
from benchmark.constructionsite10k.types import ConstructionSiteSample, SampleAttributes


def _make_sample(image_id: str, active_rules: tuple[int, ...]) -> ConstructionSiteSample:
    violations = {1: None, 2: None, 3: None, 4: None}
    for rule_id in active_rules:
        violations[rule_id] = object()  # type: ignore[assignment]
    return ConstructionSiteSample(
        image_id=image_id,
        image=None,
        image_caption=f"sample-{image_id}",
        attributes=SampleAttributes(
            illumination="normal lighting",
            camera_distance="mid distance",
            view="elevation view",
            quality_of_info="rich info",
        ),
        violations=violations,
        object_boxes={},
    )


def test_build_balanced_subset_registry_selects_clean_and_single_rule_buckets() -> None:
    """Balanced subset selection should exclude multi-violation rows and keep fixed bucket sizes."""
    samples = (
        _make_sample("c2", ()),
        _make_sample("c1", ()),
        _make_sample("r1b", (1,)),
        _make_sample("r1a", (1,)),
        _make_sample("r2b", (2,)),
        _make_sample("r2a", (2,)),
        _make_sample("r3b", (3,)),
        _make_sample("r3a", (3,)),
        _make_sample("r4b", (4,)),
        _make_sample("r4a", (4,)),
        _make_sample("multi", (1, 2)),
    )

    registry = build_balanced_subset_registry(samples, per_bucket=2, subset_name="demo")

    assert registry["demo_clean"] == ("c1", "c2")
    assert registry["demo_rule1"] == ("r1a", "r1b")
    assert registry["demo_rule2"] == ("r2a", "r2b")
    assert registry["demo_rule3"] == ("r3a", "r3b")
    assert registry["demo_rule4"] == ("r4a", "r4b")
    assert registry["demo"] == (
        "c1",
        "c2",
        "r1a",
        "r1b",
        "r2a",
        "r2b",
        "r3a",
        "r3b",
        "r4a",
        "r4b",
    )


def test_build_balanced_subset_registry_fails_when_bucket_is_too_small() -> None:
    """Subset construction should fail loudly if a bucket lacks enough samples."""
    samples = (_make_sample("only-clean", ()),)

    try:
        build_balanced_subset_registry(samples, per_bucket=1)
    except ValueError as exc:
        assert "rule1" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_balanced_subset_registry_respects_sample_filter() -> None:
    """Optional sample filters should be applied before bucket selection."""
    samples = (
        _make_sample("c1", ()),
        _make_sample("c2", ()),
        _make_sample("r1a", (1,)),
        _make_sample("r1b", (1,)),
        _make_sample("r2a", (2,)),
        _make_sample("r2b", (2,)),
        _make_sample("r3a", (3,)),
        _make_sample("r3b", (3,)),
        _make_sample("r4a", (4,)),
        _make_sample("r4b", (4,)),
    )

    registry = build_balanced_subset_registry(
        samples,
        per_bucket=1,
        subset_name="filtered",
        sample_filter=lambda sample: sample.image_id != "c1",
    )

    assert registry["filtered_clean"] == ("c2",)
