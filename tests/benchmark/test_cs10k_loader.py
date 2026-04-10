"""Tests for dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.constructionsite10k.loader import ConstructionSite10kDataset


def test_dataset_loads_samples_from_json_file(
    tmp_path: Path,
    sample_annotation: dict[str, Any],
) -> None:
    """The dataset helper should read JSON files into typed samples."""
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([sample_annotation]), encoding="utf-8")

    dataset = ConstructionSite10kDataset.from_json(path)

    assert len(dataset) == 1
    assert dataset.get_by_image_id("0000424").image_id == "0000424"
