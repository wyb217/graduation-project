"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sample_annotation() -> dict[str, Any]:
    """Return an official-style ConstructionSite10k annotation fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "constructionsite10k_sample.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture
def registry_path() -> Path:
    """Return the path to a frozen split registry fixture."""
    return Path(__file__).parent / "fixtures" / "constructionsite10k_registry.json"
