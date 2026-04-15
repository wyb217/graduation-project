"""Integration tests for the Rule 1 small-loop script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common.io.json_io import read_json, write_json
from common.schemas.bbox import NormalizedBBox
from common.schemas.point1 import Point1BaselineRecord, Point1ImagePredictionSet, Point1Prediction


def _load_rule1_module():
    module_path = Path("scripts/run_point1_rule1_pipeline.py")
    spec = importlib.util.spec_from_file_location("run_point1_rule1_script", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load scripts/run_point1_rule1_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_point1_rule1_pipeline_writes_output_and_summary(tmp_path: Path) -> None:
    """The Rule 1 script should load a frozen subset and emit a binary summary."""
    module = _load_rule1_module()
    dataset_path = tmp_path / "target.parquet"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"
    summary_path = tmp_path / "rule1.summary.json"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": "clean-1",
                    "image": {"bytes": b"fake-image-clean", "path": "clean.jpg"},
                    "image_caption": "clean sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": None,
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
                {
                    "image_id": "rule1-1",
                    "image": {"bytes": b"fake-image-rule1", "path": "rule1.jpg"},
                    "image_caption": "rule1 sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r1"},
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
            ]
        ),
        dataset_path,
    )
    write_json(
        registry_path,
        {
            "demo_clean": ["clean-1"],
            "demo_rule1": ["rule1-1"],
        },
    )

    calls: dict[str, object] = {}

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        calls["num_samples"] = len(target_samples)
        calls["has_bytes"] = all(
            sample.image is not None and sample.image.bytes is not None for sample in target_samples
        )
        calls["extra_kwargs"] = kwargs
        return [
            Point1BaselineRecord(
                image_id="clean-1",
                provider_name="rule1_pipeline",
                model_name="opencv_hog+heuristic_rule1",
                mode="rule1_smallloop",
                raw_response_text="",
                parsed_output=Point1ImagePredictionSet(
                    image_id="clean-1",
                    predictions=(
                        Point1Prediction(
                            rule_id=1,
                            decision_state="no_violation",
                            target_bbox=None,
                            supporting_evidence_ids=(),
                            counter_evidence_ids=("person-1:hard_hat_visible",),
                            unknown_items=(),
                            reason_slots={},
                            reason_text="clean",
                            confidence=0.9,
                        ),
                    ),
                ),
            ),
            Point1BaselineRecord(
                image_id="rule1-1",
                provider_name="rule1_pipeline",
                model_name="opencv_hog+heuristic_rule1",
                mode="rule1_smallloop",
                raw_response_text="",
                parsed_output=Point1ImagePredictionSet(
                    image_id="rule1-1",
                    predictions=(
                        Point1Prediction(
                            rule_id=1,
                            decision_state="violation",
                            target_bbox=NormalizedBBox(0.1, 0.2, 0.3, 0.4),
                            supporting_evidence_ids=("person-1:hard_hat_visible",),
                            counter_evidence_ids=(),
                            unknown_items=(),
                            reason_slots={},
                            reason_text="rule1",
                            confidence=0.9,
                        ),
                    ),
                ),
            ),
        ]

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
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert calls["num_samples"] == 2
    assert calls["has_bytes"] is True
    assert calls["extra_kwargs"]["provider_name"] == "rule1_pipeline"
    assert len(read_json(output_path)) == 2
    summary = read_json(summary_path)
    assert summary["rule1_precision"] == 1.0
    assert summary["rule1_recall"] == 1.0


def test_run_point1_rule1_pipeline_builds_vlm_backend_when_requested(
    tmp_path: Path,
) -> None:
    """The script should wire provider-backed VLM predicate extraction on demand."""
    module = _load_rule1_module()
    dataset_path = tmp_path / "target.parquet"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": "clean-1",
                    "image": {"bytes": b"fake-image-clean", "path": "clean.jpg"},
                    "image_caption": "clean sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": None,
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
                {
                    "image_id": "rule1-1",
                    "image": {"bytes": b"fake-image-rule1", "path": "rule1.jpg"},
                    "image_caption": "rule1 sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r1"},
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
            ]
        ),
        dataset_path,
    )
    write_json(
        registry_path,
        {
            "demo_clean": ["clean-1"],
            "demo_rule1": ["rule1-1"],
        },
    )

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

    module.load_provider_catalog = lambda path: FakeCatalog()
    module.OpenAICompatibleVisionClient = FakeClient
    module.VLMRule1PredicateExtractor = FakeExtractor
    module.Rule1Pipeline = FakePipeline
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
    module = _load_rule1_module()
    dataset_path = tmp_path / "target.parquet"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": "clean-1",
                    "image": {"bytes": b"fake-image-clean", "path": "clean.jpg"},
                    "image_caption": "clean sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": None,
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
                {
                    "image_id": "rule1-1",
                    "image": {"bytes": b"fake-image-rule1", "path": "rule1.jpg"},
                    "image_caption": "rule1 sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r1"},
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
            ]
        ),
        dataset_path,
    )
    write_json(
        registry_path,
        {
            "demo_clean": ["clean-1"],
            "demo_rule1": ["rule1-1"],
        },
    )

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

    module.LocalQwenLoadConfig = FakeLoadConfig
    module.LocalQwen3VLClient = FakeClient
    module.LocalQwenRule1PredicateExtractor = FakeExtractor
    module.Rule1Pipeline = FakePipeline
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


def test_run_point1_rule1_pipeline_builds_hybrid_candidate_backend_when_requested(
    tmp_path: Path,
) -> None:
    """The script should wire the hog_then_torchvision candidate backend on demand."""
    module = _load_rule1_module()
    dataset_path = tmp_path / "target.parquet"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": "clean-1",
                    "image": {"bytes": b"fake-image-clean", "path": "clean.jpg"},
                    "image_caption": "clean sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": None,
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
                {
                    "image_id": "rule1-1",
                    "image": {"bytes": b"fake-image-rule1", "path": "rule1.jpg"},
                    "image_caption": "rule1 sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r1"},
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
            ]
        ),
        dataset_path,
    )
    write_json(
        registry_path,
        {
            "demo_clean": ["clean-1"],
            "demo_rule1": ["rule1-1"],
        },
    )

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

    module.HogThenTorchvisionPersonCandidateGenerator = FakeCandidateGenerator
    module.Rule1Pipeline = FakePipeline
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


def test_run_point1_rule1_pipeline_supports_fulltest_mode_without_registry(
    tmp_path: Path,
) -> None:
    """The script should support dataset-backed Rule 1 fulltest summaries."""
    module = _load_rule1_module()
    dataset_path = tmp_path / "target.parquet"
    output_path = tmp_path / "rule1.json"
    summary_path = tmp_path / "rule1.summary.json"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": "clean-1",
                    "image": {"bytes": b"fake-image-clean", "path": "clean.jpg"},
                    "image_caption": "clean sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": None,
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
                {
                    "image_id": "rule1-1",
                    "image": {"bytes": b"fake-image-rule1", "path": "rule1.jpg"},
                    "image_caption": "rule1 sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r1"},
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
            ]
        ),
        dataset_path,
    )

    built: dict[str, object] = {}

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        built["num_samples"] = len(target_samples)
        return [
            Point1BaselineRecord(
                image_id="clean-1",
                provider_name="rule1_pipeline",
                model_name="opencv_hog+heuristic_rule1",
                mode="rule1_smallloop",
                raw_response_text="",
                parsed_output=Point1ImagePredictionSet(
                    image_id="clean-1",
                    predictions=(
                        Point1Prediction(
                            rule_id=1,
                            decision_state="no_violation",
                            target_bbox=None,
                            supporting_evidence_ids=(),
                            counter_evidence_ids=("person-1:hard_hat_visible",),
                            unknown_items=(),
                            reason_slots={},
                            reason_text="clean",
                            confidence=0.9,
                        ),
                    ),
                ),
            ),
            Point1BaselineRecord(
                image_id="rule1-1",
                provider_name="rule1_pipeline",
                model_name="opencv_hog+heuristic_rule1",
                mode="rule1_smallloop",
                raw_response_text="",
                parsed_output=Point1ImagePredictionSet(
                    image_id="rule1-1",
                    predictions=(
                        Point1Prediction(
                            rule_id=1,
                            decision_state="violation",
                            target_bbox=NormalizedBBox(0.1, 0.2, 0.3, 0.4),
                            supporting_evidence_ids=("person-1:hard_hat_visible",),
                            counter_evidence_ids=(),
                            unknown_items=(),
                            reason_slots={},
                            reason_text="rule1",
                            confidence=0.9,
                        ),
                    ),
                ),
            ),
        ]

    def fake_summarize_rule1_run_from_dataset(**kwargs):  # noqa: ANN003
        built["summary_kwargs"] = kwargs
        return {"rule1_precision": 1.0, "rule1_recall": 1.0}

    module.run_rule1_pipeline = fake_run_rule1_pipeline
    module.summarize_rule1_run_from_dataset = fake_summarize_rule1_run_from_dataset

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_rule1_pipeline.py",
            "--fulltest",
            "--target-parquet",
            str(dataset_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert built["num_samples"] == 2
    assert built["summary_kwargs"] == {
        "output_path": output_path,
        "target_parquet_paths": (dataset_path,),
    }
    assert read_json(summary_path)["rule1_precision"] == 1.0


def test_run_point1_rule1_pipeline_supports_bucketed_summary_with_explicit_positive_split(
    tmp_path: Path,
) -> None:
    """The script should support 65-style bucketed summaries for later BML runs."""
    module = _load_rule1_module()
    dataset_path = tmp_path / "target.parquet"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "rule1.json"
    summary_path = tmp_path / "rule1.summary.json"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": "clean-1",
                    "image": {"bytes": b"fake-image-clean", "path": "clean.jpg"},
                    "image_caption": "clean sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": None,
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
                {
                    "image_id": "rule2-1",
                    "image": {"bytes": b"fake-image-rule2", "path": "rule2.jpg"},
                    "image_caption": "rule2 sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": None,
                    "rule_2_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r2"},
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
                {
                    "image_id": "rule1-1",
                    "image": {"bytes": b"fake-image-rule1", "path": "rule1.jpg"},
                    "image_caption": "rule1 sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r1"},
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                },
            ]
        ),
        dataset_path,
    )
    write_json(
        registry_path,
        {
            "demo_clean": ["clean-1"],
            "demo_rule2": ["rule2-1"],
            "demo_rule1": ["rule1-1"],
        },
    )

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        return [
            Point1BaselineRecord(
                image_id="clean-1",
                provider_name="rule1_pipeline",
                model_name="opencv_hog+heuristic_rule1",
                mode="rule1_smallloop",
                raw_response_text="",
                parsed_output=Point1ImagePredictionSet(
                    image_id="clean-1",
                    predictions=(
                        Point1Prediction(
                            rule_id=1,
                            decision_state="violation",
                            target_bbox=NormalizedBBox(0.1, 0.2, 0.3, 0.4),
                            supporting_evidence_ids=("person-1:hard_hat_visible",),
                            counter_evidence_ids=(),
                            unknown_items=(),
                            reason_slots={},
                            reason_text="clean-fp",
                            confidence=0.9,
                        ),
                    ),
                ),
            ),
            Point1BaselineRecord(
                image_id="rule2-1",
                provider_name="rule1_pipeline",
                model_name="opencv_hog+heuristic_rule1",
                mode="rule1_smallloop",
                raw_response_text="",
                parsed_output=Point1ImagePredictionSet(
                    image_id="rule2-1",
                    predictions=(
                        Point1Prediction(
                            rule_id=1,
                            decision_state="unknown",
                            target_bbox=None,
                            supporting_evidence_ids=(),
                            counter_evidence_ids=(),
                            unknown_items=("ppe_applicable",),
                            reason_slots={},
                            reason_text="unknown",
                            confidence=0.2,
                        ),
                    ),
                ),
            ),
            Point1BaselineRecord(
                image_id="rule1-1",
                provider_name="rule1_pipeline",
                model_name="opencv_hog+heuristic_rule1",
                mode="rule1_smallloop",
                raw_response_text="",
                parsed_output=Point1ImagePredictionSet(
                    image_id="rule1-1",
                    predictions=(
                        Point1Prediction(
                            rule_id=1,
                            decision_state="violation",
                            target_bbox=NormalizedBBox(0.1, 0.2, 0.3, 0.4),
                            supporting_evidence_ids=("person-1:hard_hat_visible",),
                            counter_evidence_ids=(),
                            unknown_items=(),
                            reason_slots={},
                            reason_text="tp",
                            confidence=0.9,
                        ),
                    ),
                ),
            ),
        ]

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
            "demo_rule2",
            "demo_rule1",
            "--positive-split-name",
            "demo_rule1",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    summary = read_json(summary_path)
    assert summary["fp_by_bucket"] == {"demo_clean": 1, "demo_rule2": 0}
    assert summary["negative_hit_rate"] == 0.0


def test_run_point1_rule1_pipeline_supports_progress_checkpoint_and_failure_outputs(
    tmp_path: Path,
) -> None:
    """The script should wire optional observability outputs through the runner."""
    module = _load_rule1_module()
    dataset_path = tmp_path / "target.parquet"
    output_path = tmp_path / "rule1.json"
    summary_path = tmp_path / "rule1.summary.json"
    progress_path = tmp_path / "rule1.progress.json"
    checkpoint_path = tmp_path / "rule1.checkpoint.json"
    failure_path = tmp_path / "rule1.failures.json"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": "clean-1",
                    "image": {"bytes": b"fake-image-clean", "path": "clean.jpg"},
                    "image_caption": "clean sample",
                    "illumination": "day",
                    "camera_distance": "mid",
                    "view": "front",
                    "quality_of_info": "rich",
                    "rule_1_violation": None,
                    "rule_2_violation": None,
                    "rule_3_violation": None,
                    "rule_4_violation": None,
                }
            ]
        ),
        dataset_path,
    )

    built: dict[str, object] = {}

    def fake_run_rule1_pipeline(  # noqa: ANN001
        *,
        target_samples,
        pipeline=None,
        show_progress=False,
        **kwargs,
    ):
        built["run_kwargs"] = kwargs
        return []

    def fake_export_rule1_failures(**kwargs):  # noqa: ANN003
        built["failure_kwargs"] = kwargs
        return {
            "counts": {
                "false_positives": 0,
                "false_negatives": 0,
                "unknown_predictions": 0,
                "parse_failures": 0,
            },
            "records": [],
        }

    module.run_rule1_pipeline = fake_run_rule1_pipeline
    module.export_rule1_failures = fake_export_rule1_failures

    argv = sys.argv
    try:
        sys.argv = [
            "run_point1_rule1_pipeline.py",
            "--fulltest",
            "--target-parquet",
            str(dataset_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--progress-output",
            str(progress_path),
            "--checkpoint-output",
            str(checkpoint_path),
            "--checkpoint-every",
            "5",
            "--failure-output",
            str(failure_path),
        ]
        module.main()
    finally:
        sys.argv = argv

    assert built["run_kwargs"]["progress_output"] == progress_path
    assert built["run_kwargs"]["checkpoint_output"] == checkpoint_path
    assert built["run_kwargs"]["checkpoint_every"] == 5
    assert built["failure_kwargs"]["output_path"] == output_path
    assert failure_path.exists()
