# graduation-project

施工场景安全隐患识别硕士毕业设计代码仓库。

当前已完成的第一轮建设目标是 **Point 1 基础设施层**：

- Python `src/` 工程骨架；
- ConstructionSite10k benchmark 的 typed schema；
- annotation parser；
- JSON / parquet dataset loader；
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

推荐环境：**conda**

```bash
conda env create -f environment.yml
conda activate graduation-project
```

如果你更新了 `environment.yml`，重新同步环境可以用：

```bash
conda env update -f environment.yml --prune
```

## 最常用的日常命令

如果你不想记很多命令，直接用：

```bash
conda activate graduation-project
python scripts/dev.py all
```

它会依次执行：

1. 格式化代码
2. 检查代码
3. 运行测试

如果你只想单独运行其中一个：

```bash
python scripts/dev.py format
python scripts/dev.py check
python scripts/dev.py test
```

## 这些命令是做什么的

- `format`：自动整理代码格式
- `check`：检查明显错误和不规范写法
- `test`：运行测试，确认改动没有把已有功能弄坏

如果你 coding 基础还不强，可以把它们理解成：

- `format` = 自动排版
- `check` = 自动体检
- `test` = 自动验收

你不需要先精通这些工具，我会按这个流程帮你维持代码质量。

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

from benchmark.constructionsite10k import ConstructionSite10kDataset, SplitRegistry

registry = SplitRegistry.from_json(
    Path("src/benchmark/splits/constructionsite10k_example_registry.json")
)
dataset = ConstructionSite10kDataset.from_parquet(
    [
        Path("/Users/wyb/code/graduation-project/train-00001-of-00002.parquet"),
        Path("/Users/wyb/code/graduation-project/train-00002-of-00002.parquet"),
    ],
    registry=registry,
    split_name="train",
)
```

说明：

- `from_json(...)` 仍可用于官方风格 JSON 样例或中间产物；
- `from_parquet(...)` 用于真实 parquet shard；
- parquet 默认 **不把图片 bytes 全量读入内存**，只保留 `image.path`；
- 如后续确实需要嵌入图像字节，可传 `include_image_bytes=True`。

## 当前里程碑后的下一步

1. Point 1 baseline 接口
2. official bridge wrapper
3. Rule 1 evidence -> executor -> explanation 路径

## 快速测试子集

仓库已冻结一个快速测试子集 registry：

- `src/benchmark/splits/constructionsite10k_balanced_15x5.json`

它包含：

- clean: 15
- rule1: 15
- rule2: 15
- rule3: 15
- rule4: 15

总数是 **75**。之所以不是“60”，是因为 `4 rules + clean` 一共 5 个桶。

如需从 parquet 重新生成：

```bash
conda activate graduation-project
python scripts/build_balanced_15x5_registry.py \
  /Users/wyb/code/graduation-project/train-00001-of-00002.parquet \
  /Users/wyb/code/graduation-project/train-00002-of-00002.parquet
```

更多实现状态见：

- `docs/superpowers/specs/2026-04-10-point1-foundation-design.md`
- `docs/superpowers/plans/2026-04-10-point1-foundation.md`
- `docs/09_point1_foundation_status.md`
- `docs/11_beginner_workflow.md`
