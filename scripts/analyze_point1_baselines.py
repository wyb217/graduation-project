"""Compare direct and five-shot Point 1 baseline outputs on a subset or full dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.io.json_io import write_json
from eval.reports.point1_baseline_summary import (
    summarize_baseline_run,
    summarize_baseline_run_from_dataset,
)


def main() -> None:
    """Parse arguments and write a comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-output", type=Path, required=True)
    parser.add_argument("--few-shot-output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--subset-name", default=None)
    parser.add_argument("--target-parquet", nargs="+", type=Path, default=None)
    parser.add_argument("--target-registry", type=Path, default=None)
    parser.add_argument("--target-split", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.target_parquet is not None:
        direct_summary = summarize_baseline_run_from_dataset(
            output_path=args.direct_output,
            target_parquet_paths=tuple(args.target_parquet),
            registry_path=args.target_registry,
            split_name=args.target_split,
        )
        few_shot_summary = summarize_baseline_run_from_dataset(
            output_path=args.few_shot_output,
            target_parquet_paths=tuple(args.target_parquet),
            registry_path=args.target_registry,
            split_name=args.target_split,
        )
    else:
        if args.registry is None or args.subset_name is None:
            raise ValueError(
                "Comparison requires either --target-parquet or both --registry and --subset-name."
            )
        direct_summary = summarize_baseline_run(
            output_path=args.direct_output,
            registry_path=args.registry,
            subset_name=args.subset_name,
        )
        few_shot_summary = summarize_baseline_run(
            output_path=args.few_shot_output,
            registry_path=args.registry,
            subset_name=args.subset_name,
        )

    comparison = {
        "direct": direct_summary,
        "five_shot": few_shot_summary,
    }
    write_json(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print("\n[direct]\n" + direct_summary["rule_table_markdown"])
    print("\n[five_shot]\n" + few_shot_summary["rule_table_markdown"])


if __name__ == "__main__":
    main()
