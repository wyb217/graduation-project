"""Compare direct and five-shot Point 1 baseline outputs on a frozen subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.io.json_io import write_json
from eval.reports.point1_baseline_summary import summarize_baseline_run


def main() -> None:
    """Parse arguments and write a comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-output", type=Path, required=True)
    parser.add_argument("--few-shot-output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = {
        "direct": summarize_baseline_run(
            output_path=args.direct_output,
            registry_path=args.registry,
            subset_name=args.subset_name,
        ),
        "five_shot": summarize_baseline_run(
            output_path=args.few_shot_output,
            registry_path=args.registry,
            subset_name=args.subset_name,
        ),
    }
    write_json(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
