# graduation-project

施工场景安全隐患识别硕士毕业设计代码仓库。

当前已完成的第一轮建设目标是 **Point 1 基础设施层**：

- Python `src/` 工程骨架；
- ConstructionSite10k benchmark 的 typed schema；
- annotation parser；
- dataset loader；
- frozen split registry 读取接口；
- Point 1 结构化输出 schema；
- `ruff` / `pytest` 基础检查链路。

## 仓库边界

- `src/point1`：ConstructionSite10k 四规则闭域方法
- `src/point2`：知识增强、RAG、PEFT 方法
- `src/common`：共享 schema / IO 工具
- `src/benchmark`：数据集语义、解析、split 契约
- `src/eval`：评测与报告
- `src/system`：Point 3 系统层占位

当前实现只覆盖 Point 1 的基础层，不包含任何模型推理与外部 API。

## Quick start

要求：**Python 3.11+**

```bash
/Users/wyb/miniconda3/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest ruff
```

## 开发命令

```bash
source .venv/bin/activate
ruff format .
ruff check .
pytest -q
```

## 当前已实现的关键接口

### 1. Normalized bbox

`common.schemas.NormalizedBBox`

- 坐标格式：normalized `xyxy`
- 自动校验坐标范围与顺序

### 2. Point 1 结构化输出

`common.schemas.Point1Prediction`

包含：

- `rule_id`
- `decision_state`
- `target_bbox`
- `supporting_evidence_ids`
- `counter_evidence_ids`
- `unknown_items`
- `reason_slots`
- `reason_text`
- `confidence`

### 3. ConstructionSite10k parser / loader

```python
from pathlib import Path

from benchmark.constructionsite10k import SplitRegistry, parse_sample
from common.io.json_io import read_json

registry = SplitRegistry.from_json(
    Path("src/benchmark/splits/constructionsite10k_example_registry.json")
)
sample = parse_sample(read_json(Path("tests/fixtures/constructionsite10k_sample.json")))
```

## 当前里程碑后的下一步

1. Point 1 baseline 接口
2. official bridge wrapper
3. Rule 1 evidence -> executor -> explanation 路径

更多实现状态见：

- `docs/superpowers/specs/2026-04-10-point1-foundation-design.md`
- `docs/superpowers/plans/2026-04-10-point1-foundation.md`
- `docs/09_point1_foundation_status.md`
