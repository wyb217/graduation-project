"""Export Point 1 baseline outputs into official-style predictions and optional summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.io.json_io import read_json, write_json
from eval.bridges import export_baseline_payload_to_official_predictions
from eval.reports.point1_baseline_summary import summarize_baseline_run


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Point 1 eval helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--official-output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--subset-name", default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser


def main() -> None:
    """Parse arguments, write the official prediction export, and optionally summarize it."""
    args = build_parser().parse_args()
    baseline_payload = read_json(args.baseline_output)
    official_predictions = export_baseline_payload_to_official_predictions(baseline_payload)

    args.official_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.official_output, official_predictions)
    print(json.dumps({"official_output": str(args.official_output)}, ensure_ascii=False, indent=2))

    if args.summary_output is None:
        return
    if args.registry is None or args.subset_name is None:
        raise ValueError("--summary-output requires both --registry and --subset-name.")

    summary = summarize_baseline_run(
        output_path=args.baseline_output,
        registry_path=args.registry,
        subset_name=args.subset_name,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.summary_output, summary)
    print(json.dumps({"summary_output": str(args.summary_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
