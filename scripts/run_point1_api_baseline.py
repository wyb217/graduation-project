"""Run the Point 1 direct or 5-shot API baseline on a frozen subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.constructionsite10k.loader import ConstructionSite10kDataset
from benchmark.constructionsite10k.registry import SplitRegistry
from common.io.json_io import write_json
from point1.baselines import (
    OpenAICompatibleVisionClient,
    load_provider_catalog,
    run_api_baseline,
    select_default_five_shot_ids,
)


def main() -> None:
    """Parse arguments and run the selected baseline mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None, help="Provider name from providers.local.json.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument(
        "--mode",
        choices=("direct", "five_shot"),
        required=True,
        help="Baseline mode to run.",
    )
    parser.add_argument(
        "--target-parquet",
        nargs="+",
        type=Path,
        required=True,
        help="Parquet shard paths for the target split.",
    )
    parser.add_argument("--target-registry", type=Path, required=True)
    parser.add_argument("--target-split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    parser.add_argument("--few-shot-parquet", nargs="*", type=Path, default=[])
    parser.add_argument("--few-shot-registry", type=Path, default=None)
    parser.add_argument("--few-shot-split", default="balanced_dev_15x5")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/system/providers.local.json"),
    )
    args = parser.parse_args()

    provider_catalog = load_provider_catalog(args.config_path)
    provider = provider_catalog.get_provider(args.provider)
    model_name = provider.model if args.model is None else args.model

    target_registry = SplitRegistry.from_json(args.target_registry)
    target_dataset = ConstructionSite10kDataset.from_parquet(
        args.target_parquet,
        registry=target_registry,
        split_name=args.target_split,
        include_image_bytes=True,
    )
    target_samples = (
        target_dataset.samples if args.limit is None else target_dataset.samples[: args.limit]
    )

    example_samples = ()
    if args.mode == "five_shot":
        if args.few_shot_registry is None or not args.few_shot_parquet:
            raise ValueError("5-shot mode requires --few-shot-registry and --few-shot-parquet.")
        few_shot_registry = SplitRegistry.from_json(args.few_shot_registry)
        shot_ids = select_default_five_shot_ids(few_shot_registry, subset_name=args.few_shot_split)
        dev_dataset = ConstructionSite10kDataset.from_parquet(
            args.few_shot_parquet,
            include_image_bytes=True,
        )
        example_samples = tuple(dev_dataset.get_by_image_id(image_id) for image_id in shot_ids)

    records = run_api_baseline(
        client=OpenAICompatibleVisionClient(provider),
        model_name=model_name,
        provider_name=provider.name,
        target_samples=tuple(target_samples),
        mode=args.mode,
        example_samples=example_samples,
        show_progress=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, [record.to_dict() for record in records])
    summary_path = args.output.with_suffix(".summary.json")
    write_json(
        summary_path,
        {
            "provider_name": provider.name,
            "model_name": model_name,
            "mode": args.mode,
            "target_split": args.target_split,
            "num_records": len(records),
            "num_success": sum(record.parsed_output is not None for record in records),
            "num_failures": sum(record.parsed_output is None for record in records),
        },
    )


if __name__ == "__main__":
    main()
