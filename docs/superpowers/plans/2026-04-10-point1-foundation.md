# Point 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Point 1 milestone: a readable Python project scaffold plus ConstructionSite10k schemas, parser, loader, split registry, tests, and setup docs.

**Architecture:** Keep the first slice strictly foundational. `src/common` owns reusable contracts such as normalized bounding boxes and Point 1 output schemas; `src/benchmark/constructionsite10k` owns dataset parsing/loading/registry; `src/point1` and `src/point2` are created as boundary-preserving shells only. The implementation avoids model code and external APIs.

**Tech Stack:** Python 3.11+, setuptools, ruff, pytest, standard-library dataclasses/json/pathlib.

---

### Task 1: Scaffold the Python project and package boundaries

**Files:**
- Create: `pyproject.toml`
- Create: `src/common/__init__.py`
- Create: `src/common/io/__init__.py`
- Create: `src/common/schemas/__init__.py`
- Create: `src/benchmark/__init__.py`
- Create: `src/benchmark/constructionsite10k/__init__.py`
- Create: `src/benchmark/splits/__init__.py`
- Create: `src/eval/__init__.py`
- Create: `src/eval/reports/__init__.py`
- Create: `src/point1/__init__.py`
- Create: `src/point1/pipelines/__init__.py`
- Create: `src/point2/__init__.py`
- Create: `src/point2/pipelines/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add a failing smoke test for import boundaries**

```python
def test_point1_package_imports() -> None:
    import point1
    import point2
    import benchmark.constructionsite10k
    import common.schemas

    assert point1 is not None
    assert point2 is not None
```

- [ ] **Step 2: Run the smoke test and verify it fails**

Run: `pytest tests/conftest.py -q`
Expected: FAIL because packages and test target do not exist yet.

- [ ] **Step 3: Create the minimal project scaffold**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "graduation-project"
version = "0.1.0"
description = "Closed-domain construction safety benchmark foundation."
requires-python = ">=3.11"
dependencies = []

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

- [ ] **Step 4: Re-run the smoke test**

Run: `pytest tests/conftest.py -q`
Expected: PASS


### Task 2: Define normalized bbox and Point 1 output contracts

**Files:**
- Create: `src/common/schemas/bbox.py`
- Create: `src/common/schemas/point1.py`
- Create: `src/common/io/json_io.py`
- Create: `tests/common/test_bbox.py`
- Create: `tests/common/test_point1_schema.py`

- [ ] **Step 1: Write failing bbox tests**

```python
from common.schemas.bbox import NormalizedBBox


def test_bbox_accepts_valid_xyxy() -> None:
    bbox = NormalizedBBox(0.1, 0.2, 0.8, 0.9)
    assert bbox.to_list() == [0.1, 0.2, 0.8, 0.9]


def test_bbox_rejects_invalid_coordinate_order() -> None:
    try:
        NormalizedBBox(0.8, 0.2, 0.1, 0.9)
    except ValueError as exc:
        assert "x_min" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run the bbox tests and verify they fail**

Run: `pytest tests/common/test_bbox.py -q`
Expected: FAIL because `NormalizedBBox` is not defined yet.

- [ ] **Step 3: Implement minimal schema code**

```python
@dataclass(frozen=True, slots=True)
class NormalizedBBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        ...
```

```python
@dataclass(frozen=True, slots=True)
class Point1Evidence:
    evidence_id: str
    kind: str
    bbox: NormalizedBBox | None = None
    score: float | None = None
```

- [ ] **Step 4: Add failing tests for Point 1 structured outputs**

```python
from common.schemas.point1 import Point1Prediction


def test_point1_prediction_exposes_structured_payload() -> None:
    prediction = Point1Prediction(
        rule_id=1,
        decision_state="violation",
        target_bbox=None,
        supporting_evidence_ids=("e1",),
        counter_evidence_ids=(),
        unknown_items=("head_visibility",),
        reason_slots={"subject": "worker"},
        reason_text="worker missing hard hat",
        confidence=0.8,
    )
    assert prediction.reason_slots["subject"] == "worker"
```

- [ ] **Step 5: Run the schema tests**

Run: `pytest tests/common/test_bbox.py tests/common/test_point1_schema.py -q`
Expected: PASS


### Task 3: Implement ConstructionSite10k parser and loader

**Files:**
- Create: `src/benchmark/constructionsite10k/types.py`
- Create: `src/benchmark/constructionsite10k/parser.py`
- Create: `src/benchmark/constructionsite10k/loader.py`
- Create: `tests/fixtures/constructionsite10k_sample.json`
- Create: `tests/benchmark/test_cs10k_parser.py`
- Create: `tests/benchmark/test_cs10k_loader.py`

- [ ] **Step 1: Write failing parser tests against an official-style sample**

```python
from benchmark.constructionsite10k.parser import parse_sample


def test_parse_sample_reads_rule_and_object_boxes(sample_annotation: dict[str, object]) -> None:
    sample = parse_sample(sample_annotation)
    assert sample.image_id == "0000424"
    assert sample.violations[1] is not None
    assert sample.violations[2] is None
    assert "excavator" in sample.object_boxes
```

- [ ] **Step 2: Run the parser test and verify it fails**

Run: `pytest tests/benchmark/test_cs10k_parser.py -q`
Expected: FAIL because parser/types are not defined yet.

- [ ] **Step 3: Implement typed sample models and parser**

```python
@dataclass(frozen=True, slots=True)
class RuleViolation:
    rule_id: int
    reason: str
    bounding_boxes: tuple[NormalizedBBox, ...]
```

```python
def parse_sample(raw: Mapping[str, Any]) -> ConstructionSiteSample:
    image_id = str(raw["image_id"])
    ...
    return ConstructionSiteSample(...)
```

- [ ] **Step 4: Write failing loader tests**

```python
from benchmark.constructionsite10k.loader import ConstructionSite10kDataset


def test_dataset_loads_samples_from_json_file(tmp_path: Path, sample_annotation: dict[str, object]) -> None:
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([sample_annotation]), encoding="utf-8")

    dataset = ConstructionSite10kDataset.from_json(path)
    assert len(dataset) == 1
    assert dataset.get_by_image_id("0000424").image_id == "0000424"
```

- [ ] **Step 5: Run parser and loader tests**

Run: `pytest tests/benchmark/test_cs10k_parser.py tests/benchmark/test_cs10k_loader.py -q`
Expected: PASS


### Task 4: Implement frozen split registry support

**Files:**
- Create: `src/benchmark/constructionsite10k/registry.py`
- Create: `tests/fixtures/constructionsite10k_registry.json`
- Create: `tests/benchmark/test_cs10k_registry.py`

- [ ] **Step 1: Write failing registry tests**

```python
from benchmark.constructionsite10k.registry import SplitRegistry


def test_registry_reads_known_split(registry_path: Path) -> None:
    registry = SplitRegistry.from_json(registry_path)
    assert registry.get_split("train") == ("0000424", "0000425")


def test_registry_rejects_unknown_split(registry_path: Path) -> None:
    registry = SplitRegistry.from_json(registry_path)
    with pytest.raises(KeyError):
        registry.get_split("dev")
```

- [ ] **Step 2: Run the registry tests and verify they fail**

Run: `pytest tests/benchmark/test_cs10k_registry.py -q`
Expected: FAIL because registry support does not exist yet.

- [ ] **Step 3: Implement the registry and dataset split selection**

```python
@dataclass(frozen=True, slots=True)
class SplitRegistry:
    splits: Mapping[str, tuple[str, ...]]

    def get_split(self, split_name: str) -> tuple[str, ...]:
        ...
```

- [ ] **Step 4: Run the full benchmark test suite**

Run: `pytest tests/benchmark -q`
Expected: PASS


### Task 5: Document the scaffold and verify the project

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Create: `docs/09_point1_foundation_status.md`

- [ ] **Step 1: Update docs for setup and current milestone**

```md
## Quick start

python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest ruff

ruff check .
ruff format .
pytest -q
```

- [ ] **Step 2: Run formatting**

Run: `ruff format .`
Expected: files reformatted or left unchanged.

- [ ] **Step 3: Run lint**

Run: `ruff check .`
Expected: PASS

- [ ] **Step 4: Run all tests**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 5: Summarize remaining TODOs in docs**

```md
- Next: baseline interface
- Next: official bridge wrapper
- Next: Rule 1 evidence path
```
