"""Runtime wiring helpers for the Rule 1 runner script."""

from __future__ import annotations

import argparse

from point1.baselines import OpenAICompatibleVisionClient, load_provider_catalog
from point1.baselines.local_qwen import LocalQwen3VLClient, LocalQwenLoadConfig
from point1.candidates import HogThenTorchvisionPersonCandidateGenerator
from point1.pipelines.rule1 import Rule1Pipeline
from point1.predicates import LocalQwenRule1PredicateExtractor, VLMRule1PredicateExtractor


def build_rule1_runtime(
    args: argparse.Namespace,
) -> tuple[Rule1Pipeline | None, str, str, str]:
    """Build the Rule 1 runtime components from CLI arguments."""
    candidate_generator = build_candidate_generator(args)
    max_candidates_per_image = getattr(args, "max_candidates_per_image", None)

    if args.predicate_backend == "heuristic" and candidate_generator is None:
        return None, "rule1_pipeline", "opencv_hog+heuristic_rule1", "rule1_smallloop"
    if args.predicate_backend == "heuristic":
        pipeline = Rule1Pipeline(
            candidate_generator=candidate_generator,
            max_candidates_per_image=max_candidates_per_image,
        )
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
            max_candidates_per_image=max_candidates_per_image,
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
        candidate_batch_size=args.candidate_batch_size,
        context_mode=args.predicate_context_mode,
        crop_padding_profile=args.crop_padding_profile,
    )
    pipeline = Rule1Pipeline(
        candidate_generator=candidate_generator,
        predicate_extractor=predicate_extractor,
        max_candidates_per_image=max_candidates_per_image,
    )
    return (
        pipeline,
        "rule1_pipeline_local_qwen",
        args.model_path,
        "rule1_smallloop_local_qwen",
    )


def build_candidate_generator(args: argparse.Namespace):
    """Build the Rule 1 candidate generator from the selected backend."""
    if args.candidate_backend == "hog":
        return None
    return HogThenTorchvisionPersonCandidateGenerator(
        score_threshold=args.torchvision_score_threshold,
    )
