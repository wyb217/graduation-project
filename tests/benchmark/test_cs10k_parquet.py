"""Tests for ConstructionSite10k parquet loading."""

from __future__ import annotations

from pathlib import Path

from benchmark.constructionsite10k.loader import ConstructionSite10kDataset
from benchmark.constructionsite10k.parser import parse_sample
from benchmark.constructionsite10k.registry import SplitRegistry


def test_parse_sample_reads_embedded_image_metadata() -> None:
    """The parser should preserve embedded image metadata when present."""
    sample = parse_sample(
        {
            "image": {"bytes": b"demo-bytes", "path": "demo.jpg"},
            "image_id": "demo",
            "image_caption": "demo sample",
            "illumination": "normal lighting",
            "camera_distance": "mid distance",
            "view": "elevation view",
            "quality_of_info": "rich info",
            "rule_1_violation": None,
        }
    )

    assert sample.image is not None
    assert sample.image.path == "demo.jpg"
    assert sample.image.bytes == b"demo-bytes"


def test_dataset_loads_parquet_and_skips_image_bytes_by_default(
    parquet_dataset_path: Path,
) -> None:
    """Parquet loading should keep image paths while avoiding heavy byte payloads by default."""
    dataset = ConstructionSite10kDataset.from_parquet(parquet_dataset_path)

    sample = dataset.get_by_image_id("0000424")
    assert len(dataset) == 2
    assert sample.image is not None
    assert sample.image.path == "0000424.jpg"
    assert sample.image.bytes is None


def test_dataset_loads_parquet_with_registry_filter(
    parquet_dataset_path: Path,
    registry_path: Path,
) -> None:
    """Parquet loading should respect the frozen split registry filter."""
    registry = SplitRegistry.from_json(registry_path)

    dataset = ConstructionSite10kDataset.from_parquet(
        parquet_dataset_path,
        registry=registry,
        split_name="train",
    )

    assert tuple(sample.image_id for sample in dataset.samples) == ("0000424", "0000425")
