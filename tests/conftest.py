"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
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


@pytest.fixture
def parquet_rows(sample_annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return parquet-style rows with embedded image metadata."""
    first_row = {
        **sample_annotation,
        "image": {"bytes": b"fake-image-0000424", "path": "0000424.jpg"},
    }
    second_row = {
        **sample_annotation,
        "image_id": "0000425",
        "image_caption": "A worker stands under a crane.",
        "image": {"bytes": b"fake-image-0000425", "path": "0000425.jpg"},
        "rule_1_violation": None,
    }
    return [first_row, second_row]


@pytest.fixture
def parquet_dataset_path(tmp_path: Path, parquet_rows: list[dict[str, Any]]) -> Path:
    """Write a tiny parquet dataset for loader tests."""
    path = tmp_path / "cs10k-mini.parquet"
    table = pa.Table.from_pylist(parquet_rows)
    pq.write_table(table, path)
    return path
