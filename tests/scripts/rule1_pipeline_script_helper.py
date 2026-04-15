"""Shared helpers for Rule 1 runner script tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common.io.json_io import write_json


def load_rule1_script_module():
    """Load the Rule 1 runner script as a module for CLI-level testing."""
    module_path = Path("scripts/run_point1_rule1_pipeline.py")
    spec = importlib.util.spec_from_file_location("run_point1_rule1_script", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load scripts/run_point1_rule1_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_rule1_rows(dataset_path: Path, rows: list[dict[str, object]]) -> None:
    """Write a minimal parquet dataset for Rule 1 script tests."""
    pq.write_table(pa.Table.from_pylist(rows), dataset_path)


def build_base_row(
    *,
    image_id: str,
    image_bytes: bytes,
    image_path: str,
    rule_1_violation: dict[str, object] | None = None,
    rule_2_violation: dict[str, object] | None = None,
    rule_3_violation: dict[str, object] | None = None,
    rule_4_violation: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one minimal benchmark-style row for script tests."""
    return {
        "image_id": image_id,
        "image": {"bytes": image_bytes, "path": image_path},
        "image_caption": f"{image_id} sample",
        "illumination": "day",
        "camera_distance": "mid",
        "view": "front",
        "quality_of_info": "rich",
        "rule_1_violation": rule_1_violation,
        "rule_2_violation": rule_2_violation,
        "rule_3_violation": rule_3_violation,
        "rule_4_violation": rule_4_violation,
    }


def write_registry_payload(registry_path: Path, payload: dict[str, list[str]]) -> None:
    """Write a small split registry JSON for Rule 1 script tests."""
    write_json(registry_path, payload)
