"""Tests for frozen split registries."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.constructionsite10k.registry import SplitRegistry


def test_registry_reads_known_split(registry_path: Path) -> None:
    """Known split names should return stable image IDs."""
    registry = SplitRegistry.from_json(registry_path)

    assert registry.get_split("train") == ("0000424", "0000425")


def test_registry_rejects_unknown_split(registry_path: Path) -> None:
    """Unknown split names should fail loudly."""
    registry = SplitRegistry.from_json(registry_path)

    with pytest.raises(KeyError, match="Unknown split"):
        registry.get_split("dev")
