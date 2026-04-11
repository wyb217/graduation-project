"""Dataset loading helpers for ConstructionSite10k."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from benchmark.constructionsite10k.parser import parse_sample
from benchmark.constructionsite10k.registry import SplitRegistry
from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.io.json_io import read_json


@dataclass(frozen=True, slots=True)
class ConstructionSite10kDataset:
    """In-memory typed access to parsed ConstructionSite10k samples."""

    samples: tuple[ConstructionSiteSample, ...]

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        registry: SplitRegistry | None = None,
        split_name: str | None = None,
    ) -> ConstructionSite10kDataset:
        """Load samples from a JSON file and optionally filter by frozen split."""
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError("ConstructionSite10k annotations must be stored as a list of samples.")

        samples = tuple(parse_sample(item) for item in payload)
        dataset = cls(samples=samples)
        if registry is None or split_name is None:
            return dataset
        return dataset.select_image_ids(registry.get_split(split_name))

    @classmethod
    def from_parquet(
        cls,
        paths: Path | Iterable[Path],
        *,
        registry: SplitRegistry | None = None,
        split_name: str | None = None,
        include_image_bytes: bool = False,
    ) -> ConstructionSite10kDataset:
        """Load samples from one or more parquet shards."""
        shard_paths = _normalize_paths(paths)
        samples = []
        for path in shard_paths:
            table = pq.read_table(path)
            samples.extend(
                parse_sample(record, include_image_bytes=include_image_bytes)
                for record in table.to_pylist()
            )

        dataset = cls(samples=tuple(samples))
        if registry is None or split_name is None:
            return dataset
        return dataset.select_image_ids(registry.get_split(split_name))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def get_by_image_id(self, image_id: str) -> ConstructionSiteSample:
        """Return a sample by image ID or raise a KeyError."""
        for sample in self.samples:
            if sample.image_id == image_id:
                return sample
        raise KeyError(f"Unknown image_id: {image_id}")

    def select_image_ids(self, image_ids: Iterable[str]) -> ConstructionSite10kDataset:
        """Return a new dataset containing only the requested image IDs."""
        image_id_set = set(image_ids)
        return ConstructionSite10kDataset(
            samples=tuple(sample for sample in self.samples if sample.image_id in image_id_set)
        )


def _normalize_paths(paths: Path | Iterable[Path]) -> tuple[Path, ...]:
    """Normalize a parquet path argument into a non-empty tuple."""
    if isinstance(paths, Path):
        normalized = (paths,)
    else:
        normalized = tuple(Path(path) for path in paths)

    if not normalized:
        raise ValueError("At least one parquet path is required.")
    return normalized
