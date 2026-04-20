"""Run the Rule 1 pipeline on one raw image and export JSON plus bbox visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from common.io.json_io import write_json
from point1.candidates.person import OpenCVHogPersonCandidateGenerator
from point1.pipelines import Rule1Pipeline
from point1.pipelines.rule1_runner_runtime import (
    build_candidate_generator as _build_candidate_generator,
)
from point1.pipelines.rule1_runner_runtime import build_rule1_runtime as _build_rule1_runtime
from point1.pipelines.single_image import (
    SingleImageCandidatePrediction,
    SingleImageSource,
    build_single_image_output,
    build_single_image_sample,
    download_image_to_path,
    render_rule1_visualization,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for single-image Rule 1 runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image-path", type=Path, default=None, help="Local image path.")
    source_group.add_argument("--image-url", default=None, help="Remote image URL to download.")
    parser.add_argument(
        "--downloaded-image-path",
        type=Path,
        default=None,
        help="Optional local path for a downloaded --image-url file.",
    )
    parser.add_argument("--image-id", default=None, help="Optional explicit image id.")
    parser.add_argument("--image-caption", default=None, help="Optional explicit image caption.")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path.")
    parser.add_argument(
        "--visualization-output",
        type=Path,
        default=None,
        help="Optional annotated-image output path.",
    )
    parser.add_argument(
        "--candidate-backend",
        choices=("hog", "torchvision", "hog_then_torchvision"),
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
        help="Optional crop expansion profile for local-Qwen Rule 1 inputs.",
    )
    return parser


def main() -> None:
    """Parse arguments, run Rule 1 on one image, and write artifacts."""
    args = build_parser().parse_args()
    resolved_image_path, image_source = _resolve_image_source(args)
    sample = build_single_image_sample(
        resolved_image_path,
        image_id=image_source.image_id,
        image_caption=args.image_caption,
    )

    pipeline, provider_name, model_name, mode = _build_rule1_runtime(args)
    if pipeline is None:
        pipeline = Rule1Pipeline()
    result = pipeline.run(sample)

    candidate_generator = _build_candidate_generator(args)
    if candidate_generator is None:
        candidate_generator = OpenCVHogPersonCandidateGenerator()
    candidates = candidate_generator.generate(sample)
    if len(candidates) != len(result.candidate_predictions):
        raise ValueError(
            "candidate visualization mismatch: detector rerun did not match pipeline output size."
        )

    candidate_predictions = tuple(
        SingleImageCandidatePrediction(
            candidate_id=candidate.candidate_id,
            candidate_bbox=candidate.bbox,
            candidate_score=candidate.score,
            prediction=prediction,
        )
        for candidate, prediction in zip(candidates, result.candidate_predictions, strict=True)
    )

    visualization_output = args.visualization_output or _default_visualization_output(
        args.output,
        resolved_image_path,
    )
    render_rule1_visualization(
        image_path=resolved_image_path,
        image_source=image_source,
        candidate_predictions=candidate_predictions,
        image_prediction=result.image_prediction,
        output_path=visualization_output,
    )

    payload = build_single_image_output(
        image_source=image_source,
        provider_name=provider_name,
        model_name=model_name,
        mode=mode,
        candidate_backend=args.candidate_backend,
        predicate_backend=args.predicate_backend,
        candidate_predictions=candidate_predictions,
        image_prediction=result.image_prediction,
        prediction_set=result.to_prediction_set(),
        visualization_output=str(visualization_output),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload.to_dict())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "visualization_output": str(visualization_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _resolve_image_source(args: argparse.Namespace) -> tuple[Path, SingleImageSource]:
    if args.image_path is not None:
        resolved_path = args.image_path.resolve()
        image_id = args.image_id or resolved_path.stem
        return resolved_path, SingleImageSource(
            input_mode="path",
            local_path=str(resolved_path),
            image_id=image_id,
            original_path=str(resolved_path),
        )

    output_path = args.downloaded_image_path or _default_download_path(args.image_url, args.output)
    resolved_path = download_image_to_path(args.image_url, output_path).resolve()
    image_id = args.image_id or resolved_path.stem
    return resolved_path, SingleImageSource(
        input_mode="url",
        local_path=str(resolved_path),
        image_id=image_id,
        original_url=args.image_url,
    )


def _default_download_path(image_url: str, output_path: Path) -> Path:
    parsed = urlparse(image_url)
    suffix = Path(parsed.path).suffix or ".jpg"
    return output_path.with_suffix(f".downloaded{suffix}")


def _default_visualization_output(output_path: Path, image_path: Path) -> Path:
    suffix = image_path.suffix or ".jpg"
    return output_path.with_suffix(f".visualized{suffix}")


if __name__ == "__main__":
    main()
