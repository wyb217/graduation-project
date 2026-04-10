"""Build the frozen balanced_15x5 registry from ConstructionSite10k parquet shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.constructionsite10k.loader import ConstructionSite10kDataset
from benchmark.constructionsite10k.subsets import build_balanced_subset_registry
from common.io.json_io import write_json


def main() -> None:
    """Parse arguments and export the balanced quick-test subset registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", type=Path, help="Input parquet shard paths.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/benchmark/splits/constructionsite10k_balanced_15x5.json"),
        help="Output registry JSON path.",
    )
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=15,
        help="Number of samples to keep for clean and each single-rule bucket.",
    )
    args = parser.parse_args()

    dataset = ConstructionSite10kDataset.from_parquet(args.parquet)
    registry = build_balanced_subset_registry(dataset.samples, per_bucket=args.per_bucket)
    write_json(
        args.output,
        {split_name: list(image_ids) for split_name, image_ids in registry.items()},
    )


if __name__ == "__main__":
    main()
