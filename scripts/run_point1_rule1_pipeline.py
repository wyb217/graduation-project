"""Run the Rule 1 small closed-loop pipeline on a frozen clean/rule1 subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.constructionsite10k.loader import ConstructionSite10kDataset
from benchmark.constructionsite10k.registry import SplitRegistry
from common.io.json_io import write_json
from eval.reports.point1_rule1_summary import (
    summarize_rule1_bucketed_run,
    summarize_rule1_smallloop,
)
from point1.baselines import OpenAICompatibleVisionClient, load_provider_catalog
from point1.baselines.local_qwen import LocalQwen3VLClient, LocalQwenLoadConfig
from point1.candidates import HogThenTorchvisionPersonCandidateGenerator
from point1.pipelines import run_rule1_pipeline
from point1.pipelines.rule1 import Rule1Pipeline
from point1.predicates import LocalQwenRule1PredicateExtractor, VLMRule1PredicateExtractor


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Rule 1 small-loop runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-parquet",
        nargs="+",
        type=Path,
        required=True,
        help="Parquet shard paths for the target dataset.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Frozen registry containing the clean/rule1 split names to run.",
    )
    parser.add_argument(
        "--target-split-names",
        nargs="+",
        required=True,
        help=(
            "Ordered split names to combine, e.g. clean then rule1 or "
            "clean/rule2/rule3/rule4/rule1."
        ),
    )
    parser.add_argument(
        "--positive-split-name",
        default=None,
        help="Explicit positive Rule 1 split name. Required for multi-bucket runs with 3+ splits.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-backend",
        choices=("hog", "hog_then_torchvision"),
        default="hog",
        help="Candidate generator backend for Rule 1.",
    )
    parser.add_argument(
        "--torchvision-score-threshold",
        type=float,
        default=0.3,
        help="Score threshold for the torchvision fallback detector.",
    )
    parser.add_argument(
        "--predicate-backend",
        choices=("heuristic", "vlm", "local_qwen"),
        default="heuristic",
        help="Predicate extractor backend for Rule 1.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider name from providers.local.json when --predicate-backend=vlm.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional VLM model override when --predicate-backend=vlm.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local Qwen model path when --predicate-backend=local_qwen.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for local Qwen loading when --predicate-backend=local_qwen.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Generation cap for local Qwen predicate extraction.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation for local Qwen loading.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/system/providers.local.json"),
        help="Provider config path for VLM predicate extraction.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional explicit path for the Rule 1 binary summary JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs after split filtering.",
    )
    return parser


def main() -> None:
    """Parse arguments, run the Rule 1 pipeline, and write output plus summary."""
    args = build_parser().parse_args()

    if len(args.target_split_names) < 2:
        raise ValueError("Rule 1 run expects at least two split names.")

    registry = SplitRegistry.from_json(args.registry)
    positive_split_name = _resolve_positive_split_name(args)
    target_image_ids = _merge_split_image_ids(registry, tuple(args.target_split_names))
    dataset = ConstructionSite10kDataset.from_parquet(
        args.target_parquet,
        include_image_bytes=True,
        image_ids=target_image_ids,
    )
    available_sample_ids = {sample.image_id for sample in dataset.samples}
    ordered_samples = tuple(
        dataset.get_by_image_id(image_id)
        for image_id in target_image_ids
        if image_id in available_sample_ids
    )
    target_samples = ordered_samples if args.limit is None else ordered_samples[: args.limit]
    pipeline, provider_name, model_name, mode = _build_rule1_runtime(args)

    records = run_rule1_pipeline(
        target_samples=target_samples,
        pipeline=pipeline,
        provider_name=provider_name,
        model_name=model_name,
        mode=mode,
        show_progress=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, [record.to_dict() for record in records])
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False, indent=2))

    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    if (
        args.limit is None
        and len(args.target_split_names) == 2
        and positive_split_name == args.target_split_names[1]
    ):
        summary = summarize_rule1_smallloop(
            output_path=args.output,
            registry_path=args.registry,
            clean_split_name=args.target_split_names[0],
            positive_split_name=positive_split_name,
        )
    else:
        selected_image_ids = tuple(sample.image_id for sample in target_samples)
        selected_image_id_set = set(selected_image_ids)
        bucket_image_ids = {
            split_name: tuple(
                image_id
                for image_id in registry.get_split(split_name)
                if image_id in selected_image_id_set
            )
            for split_name in args.target_split_names
        }
        summary = summarize_rule1_bucketed_run(
            output_path=args.output,
            bucket_image_ids=bucket_image_ids,
            positive_bucket_name=positive_split_name,
        )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_output, summary)
    print(json.dumps({"summary_output": str(summary_output)}, ensure_ascii=False, indent=2))


def _merge_split_image_ids(
    registry: SplitRegistry,
    split_names: tuple[str, ...],
) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered_ids: list[str] = []
    for split_name in split_names:
        for image_id in registry.get_split(split_name):
            if image_id in seen:
                continue
            seen.add(image_id)
            ordered_ids.append(image_id)
    return tuple(ordered_ids)


def _build_rule1_runtime(
    args: argparse.Namespace,
) -> tuple[Rule1Pipeline | None, str, str, str]:
    candidate_generator = _build_candidate_generator(args)

    if args.predicate_backend == "heuristic" and candidate_generator is None:
        return None, "rule1_pipeline", "opencv_hog+heuristic_rule1", "rule1_smallloop"
    if args.predicate_backend == "heuristic":
        pipeline = Rule1Pipeline(candidate_generator=candidate_generator)
        return pipeline, "rule1_pipeline", "opencv_hog+heuristic_rule1", "rule1_smallloop"

    if args.predicate_backend == "vlm":
        provider_catalog = load_provider_catalog(args.config_path)
        provider = provider_catalog.get_provider(args.provider)
        model_name = provider.model if args.model is None else args.model
        client = OpenAICompatibleVisionClient(provider)
        predicate_extractor = VLMRule1PredicateExtractor(
            client=client,
            model_name=model_name,
            provider_name=provider.name,
        )
        pipeline = Rule1Pipeline(
            candidate_generator=candidate_generator,
            predicate_extractor=predicate_extractor,
        )
        return (
            pipeline,
            f"rule1_pipeline_{provider.name}",
            model_name,
            "rule1_smallloop_vlm",
        )

    if not args.model_path:
        raise ValueError("--model-path is required when --predicate-backend=local_qwen.")

    client = LocalQwen3VLClient(
        LocalQwenLoadConfig(
            model_path=args.model_path,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
        )
    )
    predicate_extractor = LocalQwenRule1PredicateExtractor(
        client=client,
        model_name=args.model_path,
    )
    pipeline = Rule1Pipeline(
        candidate_generator=candidate_generator,
        predicate_extractor=predicate_extractor,
    )
    return (
        pipeline,
        "rule1_pipeline_local_qwen",
        args.model_path,
        "rule1_smallloop_local_qwen",
    )


def _build_candidate_generator(args: argparse.Namespace):
    if args.candidate_backend == "hog":
        return None
    return HogThenTorchvisionPersonCandidateGenerator(
        score_threshold=args.torchvision_score_threshold,
    )


def _resolve_positive_split_name(args: argparse.Namespace) -> str:
    if args.positive_split_name is not None:
        if args.positive_split_name not in args.target_split_names:
            raise ValueError("--positive-split-name must be one of --target-split-names.")
        return str(args.positive_split_name)
    if len(args.target_split_names) == 2:
        return str(args.target_split_names[1])
    raise ValueError("Multi-bucket runs require --positive-split-name.")


if __name__ == "__main__":
    main()
