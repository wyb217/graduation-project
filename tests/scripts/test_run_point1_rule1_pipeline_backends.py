"""Backend wiring tests for the Rule 1 runner script."""

from __future__ import annotations

import sys
from pathlib import Path

from point1.pipelines import rule1_runner_runtime as runtime_module
from tests.scripts.rule1_pipeline_script_helper import (
    build_base_row,
    load_rule1_script_module,
    write_registry_payload,
    write_rule1_rows,
)


def _write_clean_rule1_fixture(dataset_path: Path, registry_path: Path) -> None:
    write_rule1_rows(
        dataset_path,
        [
            build_base_row(
                image_id="clean-1",
                image_bytes=b"fake-image-clean",
                image_path="clean.jpg",
            ),
            build_base_row(
                image_id="rule1-1",
                image_bytes=b"fake-image-rule1",
                image_path="rule1.jpg",
                rule_1_violation={"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r1"},
            ),
        ],
    )
    write_registry_payload(
        registry_path,
        {
            "demo_clean": ["clean-1"],
            "demo_rule1": ["rule1-1"],
        },
    )


def test_run_point1_rule1_pipeline_builds_vlm_backend_when_requested(
    tmp_path: Path,
) -> None:
    """The script should wire provider-backed VLM predicate extraction on demand."""
    module = load_rule1_script_module()
    dataset_path = tmp_path / "target.parquet"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"
    _write_clean_rule1_fixture(dataset_path, registry_path)

    built: dict[str, object] = {}

    class FakeProvider:
        name = "modelscope"
        base_url = "https://example.com"
        api_key = "secret"
        model = "demo-model"

    class FakeCatalog:
        def get_provider(self, provider_name):  # noqa: ANN001
            built["provider_name_arg"] = provider_name
            return FakeProvider()

    class FakeClient:
        def __init__(self, provider) -> None:  # noqa: ANN001
            built["provider_name"] = provider.name

    class FakeExtractor:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            built["extractor_kwargs"] = kwargs

    class FakePipeline:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            built["pipeline_kwargs"] = kwargs

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        built["run_pipeline_type"] = type(pipeline).__name__
        built["run_kwargs"] = kwargs
        return []

    runtime_module.load_provider_catalog = lambda path: FakeCatalog()
    runtime_module.OpenAICompatibleVisionClient = FakeClient
    runtime_module.VLMRule1PredicateExtractor = FakeExtractor
    runtime_module.Rule1Pipeline = FakePipeline
    module.run_rule1_pipeline = fake_run_rule1_pipeline

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_rule1_pipeline.py",
            "--target-parquet",
            str(dataset_path),
            "--registry",
            str(registry_path),
            "--target-split-names",
            "demo_clean",
            "demo_rule1",
            "--predicate-backend",
            "vlm",
            "--provider",
            "modelscope",
            "--output",
            str(output_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert built["provider_name_arg"] == "modelscope"
    assert built["provider_name"] == "modelscope"
    assert built["run_pipeline_type"] == "FakePipeline"
    assert built["run_kwargs"]["provider_name"] == "rule1_pipeline_modelscope"


def test_run_point1_rule1_pipeline_builds_local_qwen_backend_when_requested(
    tmp_path: Path,
) -> None:
    """The script should wire local Qwen predicate extraction on demand."""
    module = load_rule1_script_module()
    dataset_path = tmp_path / "target.parquet"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"
    _write_clean_rule1_fixture(dataset_path, registry_path)

    built: dict[str, object] = {}

    class FakeLoadConfig:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            built["load_config_kwargs"] = kwargs

    class FakeClient:
        def __init__(self, config) -> None:  # noqa: ANN001
            built["client_config"] = config

    class FakeExtractor:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            built["extractor_kwargs"] = kwargs

    class FakePipeline:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            built["pipeline_kwargs"] = kwargs

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        built["run_pipeline_type"] = type(pipeline).__name__
        built["run_kwargs"] = kwargs
        return []

    runtime_module.LocalQwenLoadConfig = FakeLoadConfig
    runtime_module.LocalQwen3VLClient = FakeClient
    runtime_module.LocalQwenRule1PredicateExtractor = FakeExtractor
    runtime_module.Rule1Pipeline = FakePipeline
    module.run_rule1_pipeline = fake_run_rule1_pipeline

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_rule1_pipeline.py",
            "--target-parquet",
            str(dataset_path),
            "--registry",
            str(registry_path),
            "--target-split-names",
            "demo_clean",
            "demo_rule1",
            "--predicate-backend",
            "local_qwen",
            "--model-path",
            "/models/qwen3-vl",
            "--torch-dtype",
            "bfloat16",
            "--max-new-tokens",
            "256",
            "--attn-implementation",
            "flash_attention_2",
            "--candidate-batch-size",
            "8",
            "--predicate-context-mode",
            "crop_with_full_image",
            "--crop-padding-profile",
            "rule1_ppe",
            "--output",
            str(output_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert built["load_config_kwargs"] == {
        "model_path": "/models/qwen3-vl",
        "torch_dtype": "bfloat16",
        "max_new_tokens": 256,
        "attn_implementation": "flash_attention_2",
    }
    assert built["run_pipeline_type"] == "FakePipeline"
    assert built["run_kwargs"]["provider_name"] == "rule1_pipeline_local_qwen"
    assert built["run_kwargs"]["model_name"] == "/models/qwen3-vl"
    assert built["extractor_kwargs"]["candidate_batch_size"] == 8
    assert built["extractor_kwargs"]["context_mode"] == "crop_with_full_image"
    assert built["extractor_kwargs"]["crop_padding_profile"] == "rule1_ppe"


def test_run_point1_rule1_pipeline_defaults_local_qwen_max_new_tokens_to_256(
    tmp_path: Path,
) -> None:
    """The local-Qwen backend should default to the reduced generation cap."""
    module = load_rule1_script_module()
    dataset_path = tmp_path / "target.parquet"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"
    _write_clean_rule1_fixture(dataset_path, registry_path)

    built: dict[str, object] = {}

    class FakeLoadConfig:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            built["load_config_kwargs"] = kwargs

    class FakeClient:
        def __init__(self, config) -> None:  # noqa: ANN001
            built["client_config"] = config

    class FakeExtractor:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            built["extractor_kwargs"] = kwargs

    class FakePipeline:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            built["pipeline_kwargs"] = kwargs

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        built["run_kwargs"] = kwargs
        return []

    runtime_module.LocalQwenLoadConfig = FakeLoadConfig
    runtime_module.LocalQwen3VLClient = FakeClient
    runtime_module.LocalQwenRule1PredicateExtractor = FakeExtractor
    runtime_module.Rule1Pipeline = FakePipeline
    module.run_rule1_pipeline = fake_run_rule1_pipeline

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_rule1_pipeline.py",
            "--target-parquet",
            str(dataset_path),
            "--registry",
            str(registry_path),
            "--target-split-names",
            "demo_clean",
            "demo_rule1",
            "--predicate-backend",
            "local_qwen",
            "--model-path",
            "/models/qwen3-vl",
            "--output",
            str(output_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert built["load_config_kwargs"]["max_new_tokens"] == 256


def test_run_point1_rule1_pipeline_builds_hybrid_candidate_backend_when_requested(
    tmp_path: Path,
) -> None:
    """The script should wire the hog_then_torchvision candidate backend on demand."""
    module = load_rule1_script_module()
    dataset_path = tmp_path / "target.parquet"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"
    _write_clean_rule1_fixture(dataset_path, registry_path)

    built: dict[str, object] = {}

    class FakeCandidateGenerator:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            built["candidate_kwargs"] = kwargs

    class FakePipeline:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            built["pipeline_kwargs"] = kwargs

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        built["run_pipeline_type"] = type(pipeline).__name__
        built["run_kwargs"] = kwargs
        return []

    runtime_module.HogThenTorchvisionPersonCandidateGenerator = FakeCandidateGenerator
    runtime_module.Rule1Pipeline = FakePipeline
    module.run_rule1_pipeline = fake_run_rule1_pipeline

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_rule1_pipeline.py",
            "--target-parquet",
            str(dataset_path),
            "--registry",
            str(registry_path),
            "--target-split-names",
            "demo_clean",
            "demo_rule1",
            "--candidate-backend",
            "hog_then_torchvision",
            "--torchvision-score-threshold",
            "0.42",
            "--output",
            str(output_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert built["candidate_kwargs"] == {"score_threshold": 0.42}
    assert built["run_pipeline_type"] == "FakePipeline"
    assert "candidate_generator" in built["pipeline_kwargs"]
