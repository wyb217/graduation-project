"""CLI helpers for the Rule 1 runner script."""

from __future__ import annotations

import argparse
from pathlib import Path

BALANCED65_REGISTRY_PATH = Path("src/benchmark/splits/constructionsite10k_balanced_test_13x5.json")
BALANCED65_SPLIT_NAMES = (
    "balanced_test_13x5_clean",
    "balanced_test_13x5_rule2",
    "balanced_test_13x5_rule3",
    "balanced_test_13x5_rule4",
    "balanced_test_13x5_rule1",
)
BALANCED65_POSITIVE_SPLIT = "balanced_test_13x5_rule1"


def build_rule1_runner_parser(*, description: str) -> argparse.ArgumentParser:
    """Build the CLI parser for the Rule 1 small-loop runner."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--target-parquet",
        nargs="+",
        type=Path,
        required=True,
        help="Parquet shard paths for the target dataset.",
    )
    parser.add_argument(
        "--target-preset",
        choices=("balanced65", "fulltest"),
        default=None,
        help="Optional short preset for the common balanced65 or fulltest target setup.",
    )
    parser.add_argument(
        "--fulltest",
        action="store_true",
        help="Run Rule 1 over the full target parquet without registry-driven subset filtering.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Frozen registry containing the clean/rule1 split names to run.",
    )
    parser.add_argument(
        "--target-split-names",
        nargs="+",
        default=None,
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
    parser.add_argument("--output", type=Path, required=False, default=None)
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
        default=256,
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
        "--progress-output",
        type=Path,
        default=None,
        help="Optional path for a JSON heartbeat file updated after each image.",
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=None,
        help="Optional path for writing partial baseline records during long runs.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write partial results every N images when --checkpoint-output is set.",
    )
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=None,
        help="Optional path for a Rule 1 FP/FN/unknown export JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs after split filtering.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=1,
        help="Batch size for local-Qwen predicate extraction across candidates from one image.",
    )
    parser.add_argument(
        "--predicate-context-mode",
        choices=("crop_only", "crop_with_full_image"),
        default="crop_only",
        help="Whether local-Qwen predicate prompts should include the full image context.",
    )
    parser.add_argument(
        "--crop-padding-profile",
        choices=("none", "rule1_ppe"),
        default="none",
        help="Optional crop expansion profile for the local-Qwen Rule 1 predicate input.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional short run name used to auto-generate the standard artifact paths.",
    )
    return parser


def apply_target_preset(args: argparse.Namespace) -> None:
    """Expand a short target preset into the verbose target selection arguments."""
    if args.target_preset is None:
        return

    if args.target_preset == "fulltest":
        if args.registry is not None or args.target_split_names is not None:
            raise ValueError("--target-preset fulltest cannot be combined with subset split args.")
        args.fulltest = True
        return

    if (
        args.fulltest
        or args.registry is not None
        or args.target_split_names is not None
        or args.positive_split_name is not None
    ):
        raise ValueError(
            "--target-preset balanced65 cannot be combined with explicit split selection args."
        )

    args.registry = BALANCED65_REGISTRY_PATH
    args.target_split_names = list(BALANCED65_SPLIT_NAMES)
    args.positive_split_name = BALANCED65_POSITIVE_SPLIT


def apply_run_name_defaults(args: argparse.Namespace) -> None:
    """Fill the standard artifact paths from a short run name when requested."""
    if args.run_name is None:
        if args.output is None:
            raise ValueError("--output is required unless --run-name is provided.")
        return

    artifact_dir = Path("artifacts/point1")
    stem = build_run_stem(args)

    if args.output is None:
        args.output = artifact_dir / f"{stem}.json"
    if args.summary_output is None:
        args.summary_output = artifact_dir / f"{stem}.summary.json"
    if args.progress_output is None:
        args.progress_output = artifact_dir / f"{stem}.progress.json"
    checkpoint_was_implicit = args.checkpoint_output is None
    if checkpoint_was_implicit:
        args.checkpoint_output = artifact_dir / f"{stem}.checkpoint.json"
    if args.failure_output is None:
        args.failure_output = artifact_dir / f"{stem}.failures.json"
    if checkpoint_was_implicit and args.checkpoint_every == 0:
        args.checkpoint_every = 100 if resolve_target_tag(args) == "fulltest" else 20


def build_run_stem(args: argparse.Namespace) -> str:
    """Return the default artifact stem for one Rule 1 run."""
    parts = ["rule1-smallloop"]
    if args.predicate_backend == "local_qwen":
        parts.append("localqwen")
    elif args.predicate_backend == "vlm":
        parts.extend(["vlm", args.provider or "provider"])
    elif args.predicate_backend == "heuristic":
        parts.append("heuristic")
    if args.candidate_backend == "hog_then_torchvision":
        parts.append("hybriddet")
    parts.append(resolve_target_tag(args))
    parts.append(str(args.run_name))
    return "-".join(parts)


def resolve_target_tag(args: argparse.Namespace) -> str:
    """Return a short label for the selected target family."""
    if args.target_preset == "balanced65":
        return "balanced65"
    if args.target_preset == "fulltest" or args.fulltest:
        return "fulltest"
    return "custom"
