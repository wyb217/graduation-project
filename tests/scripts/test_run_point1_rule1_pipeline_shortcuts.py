"""Preset and run-name shortcut tests for the Rule 1 runner script."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

from tests.scripts.rule1_pipeline_script_helper import load_rule1_script_module


def test_apply_target_preset_supports_balanced65_shortcut() -> None:
    """The script should expose a stable balanced65 preset for BML-friendly use."""
    module = load_rule1_script_module()
    args = Namespace(
        target_preset="balanced65",
        fulltest=False,
        registry=None,
        target_split_names=None,
        positive_split_name=None,
    )

    module._apply_target_preset(args)  # noqa: SLF001

    assert args.registry == Path("src/benchmark/splits/constructionsite10k_balanced_test_13x5.json")
    assert args.target_split_names == [
        "balanced_test_13x5_clean",
        "balanced_test_13x5_rule2",
        "balanced_test_13x5_rule3",
        "balanced_test_13x5_rule4",
        "balanced_test_13x5_rule1",
    ]
    assert args.positive_split_name == "balanced_test_13x5_rule1"


def test_apply_run_name_defaults_generates_rule1_artifacts() -> None:
    """Run names should auto-expand into the usual artifact bundle paths."""
    module = load_rule1_script_module()
    args = Namespace(
        run_name="rule1ppe",
        target_preset="balanced65",
        fulltest=False,
        candidate_backend="hog_then_torchvision",
        predicate_backend="local_qwen",
        provider=None,
        output=None,
        summary_output=None,
        progress_output=None,
        checkpoint_output=None,
        checkpoint_every=0,
        failure_output=None,
    )

    module._apply_run_name_defaults(args)  # noqa: SLF001

    stem = "rule1-smallloop-localqwen-hybriddet-balanced65-rule1ppe"
    assert args.output == Path(f"artifacts/point1/{stem}.json")
    assert args.summary_output == Path(f"artifacts/point1/{stem}.summary.json")
    assert args.progress_output == Path(f"artifacts/point1/{stem}.progress.json")
    assert args.checkpoint_output == Path(f"artifacts/point1/{stem}.checkpoint.json")
    assert args.failure_output == Path(f"artifacts/point1/{stem}.failures.json")
    assert args.checkpoint_every == 20


def test_run_point1_rule1_pipeline_supports_target_preset_and_run_name(
    tmp_path: Path,
) -> None:
    """The script should accept the short balanced65 + run-name surface."""
    module = load_rule1_script_module()
    dataset_path = tmp_path / "target.parquet"
    output_stem = "rule1-smallloop-localqwen-hybriddet-balanced65-stable"
    built: dict[str, object] = {}

    def fake_load_target_samples(args):  # noqa: ANN001
        built["registry"] = args.registry
        built["target_split_names"] = args.target_split_names
        built["positive_split_name"] = args.positive_split_name
        return (), {"mode": "fulltest"}

    def fake_build_rule1_runtime(args):  # noqa: ANN001
        built["crop_padding_profile"] = args.crop_padding_profile
        return (
            None,
            "rule1_pipeline_local_qwen",
            "/models/qwen3-vl",
            "rule1_smallloop_local_qwen",
        )

    def fake_run_rule1_pipeline(**kwargs):  # noqa: ANN003
        built["run_kwargs"] = kwargs
        return []

    def fake_build_summary(**kwargs):  # noqa: ANN003
        return {"rule1_precision": 1.0}

    module._load_target_samples = fake_load_target_samples  # noqa: SLF001
    module._build_rule1_runtime = fake_build_rule1_runtime  # noqa: SLF001
    module.run_rule1_pipeline = fake_run_rule1_pipeline
    module._build_summary = fake_build_summary  # noqa: SLF001

    argv = sys.argv
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        sys.argv = [
            "run_point1_rule1_pipeline.py",
            "--target-parquet",
            str(dataset_path),
            "--target-preset",
            "balanced65",
            "--candidate-backend",
            "hog_then_torchvision",
            "--predicate-backend",
            "local_qwen",
            "--model-path",
            "/models/qwen3-vl",
            "--run-name",
            "stable",
        ]
        module.main()
    finally:
        os.chdir(cwd)
        sys.argv = argv

    assert built["registry"] == Path(
        "src/benchmark/splits/constructionsite10k_balanced_test_13x5.json"
    )
    assert built["target_split_names"][0] == "balanced_test_13x5_clean"
    assert built["positive_split_name"] == "balanced_test_13x5_rule1"
    assert built["crop_padding_profile"] == "none"
    assert built["run_kwargs"]["progress_output"] == Path(
        f"artifacts/point1/{output_stem}.progress.json"
    )
    assert (tmp_path / f"artifacts/point1/{output_stem}.json").exists()
    assert (tmp_path / f"artifacts/point1/{output_stem}.summary.json").exists()
