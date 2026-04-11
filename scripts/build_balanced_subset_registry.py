"""Build a frozen balanced subset registry from ConstructionSite10k parquet shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.constructionsite10k.loader import ConstructionSite10kDataset
from benchmark.constructionsite10k.subsets import (
    build_balanced_subset_registry,
    make_max_image_side_filter,
)
from common.io.json_io import write_json


def main() -> None:
    """Parse arguments and export a balanced subset registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", type=Path, help="Input parquet shard paths.")
    parser.add_argument("--subset-name", required=True, help="Name prefix for the frozen subset.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output registry JSON path.",
    )
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=15,
        help="Number of samples to keep for clean and each single-rule bucket.",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=None,
        help="Optional maximum image width/height. Useful for provider image-size limits.",
    )
    args = parser.parse_args()

    dataset = ConstructionSite10kDataset.from_parquet(
        args.parquet,
        include_image_bytes=args.max_image_side is not None,
    )
    sample_filter = None
    if args.max_image_side is not None:
        sample_filter = make_max_image_side_filter(max_image_side=args.max_image_side)
    registry = build_balanced_subset_registry(
        dataset.samples,
        per_bucket=args.per_bucket,
        subset_name=args.subset_name,
        sample_filter=sample_filter,
    )
    write_json(
        args.output,
        {split_name: list(image_ids) for split_name, image_ids in registry.items()},
    )


if __name__ == "__main__":
    main()
