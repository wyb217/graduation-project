"""Frozen split registry support for ConstructionSite10k experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.io.json_io import read_json


@dataclass(frozen=True, slots=True)
class SplitRegistry:
    """A stable mapping from split names to frozen image ID tuples."""

    splits: dict[str, tuple[str, ...]]

    @classmethod
    def from_json(cls, path: Path) -> SplitRegistry:
        """Load a split registry from JSON."""
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(
                "Split registry JSON must be an object mapping split names to image IDs."
            )

        splits: dict[str, tuple[str, ...]] = {}
        for split_name, image_ids in payload.items():
            if not isinstance(split_name, str) or not isinstance(image_ids, list):
                raise ValueError(
                    "Split registry entries must map string names to lists of image IDs."
                )
            splits[split_name] = tuple(str(image_id) for image_id in image_ids)
        return cls(splits=splits)

    def get_split(self, split_name: str) -> tuple[str, ...]:
        """Return image IDs for a known split name."""
        try:
            return self.splits[split_name]
        except KeyError as exc:
            raise KeyError(f"Unknown split: {split_name}") from exc
