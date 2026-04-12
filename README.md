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

当前正在进入 **Point 1 API baseline** 阶段：

- direct / zero-shot baseline
- 5-shot baseline
- 优先走 OpenAI-compatible API provider

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

## baseline 配置

第一次运行 API baseline 前，需要准备本地 provider 配置：

```bash
cp configs/system/providers.example.json configs/system/providers.local.json
```

然后把你自己的 provider 信息写进 `configs/system/providers.local.json`。

当前代码支持 OpenAI-compatible provider 配置，优先推荐：

1. `modelscope`
2. `dashscope`
3. 其他兼容 provider

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

1. official eval bridge wrapper
2. Rule 1 evidence -> executor -> explanation 路径
3. Rule 4 pair reasoning

## 快速测试子集

仓库已冻结一个快速测试子集 registry：

- `src/benchmark/splits/constructionsite10k_balanced_dev_15x5.json`
- `src/benchmark/splits/constructionsite10k_balanced_test_13x5.json`

说明：

- `balanced_dev_15x5` 来自 train，用于调 prompt / 固定 5-shot 示例
- `balanced_test_13x5` 来自 test，用于快速评估
- test 中满足“rule4 单违规”的纯净样本只有 13 张，所以 test 侧无法构造 15x5

每个子集都只保留：

- clean
- 单规则 rule1
- 单规则 rule2
- 单规则 rule3
- 单规则 rule4

如需从 parquet 重新生成 dev 子集：

```bash
conda activate graduation-project
python scripts/build_balanced_subset_registry.py \
  train-00001-of-00002.parquet \
  train-00002-of-00002.parquet \
  --subset-name balanced_dev_15x5 \
  --output src/benchmark/splits/constructionsite10k_balanced_dev_15x5.json
```

如需从 parquet 重新生成 test 子集：

```bash
conda activate graduation-project
python scripts/build_balanced_subset_registry.py \
  test.parquet \
  --subset-name balanced_test_13x5 \
  --per-bucket 13 \
  --output src/benchmark/splits/constructionsite10k_balanced_test_13x5.json
```

## baseline 运行命令

### 1. direct / zero-shot

```bash
conda activate graduation-project
python scripts/run_point1_api_baseline.py \
  --provider modelscope \
  --mode direct \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --output artifacts/point1/direct-modelscope-balanced_test_13x5.json
```

### 2. 5-shot

```bash
conda activate graduation-project
python scripts/run_point1_api_baseline.py \
  --provider modelscope \
  --mode five_shot \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --few-shot-parquet train-00001-of-00002.parquet train-00002-of-00002.parquet \
  --few-shot-registry src/benchmark/splits/constructionsite10k_balanced_dev_15x5.json \
  --few-shot-split balanced_dev_15x5 \
  --output artifacts/point1/fiveshot-modelscope-balanced_test_13x5.json
```

## official eval bridge

如果你已经有 Point 1 baseline 输出文件，可以把它导出成 ConstructionSite10k 官方风格预测格式：

```bash
conda activate graduation-project
python scripts/run_point1_eval.py \
  --baseline-output artifacts/point1/fiveshot-modelscope-balanced_test_13x5.json \
  --official-output artifacts/point1/fiveshot-modelscope-balanced_test_13x5.official.json \
  --registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --subset-name balanced_test_13x5 \
  --summary-output artifacts/point1/fiveshot-modelscope-balanced_test_13x5.eval-summary.json
```

这个脚本会：

- 导出官方风格预测文件，便于后续对接官方评测仓库；
- 可选生成当前仓库内部 summary，先看 parse success、Macro-F1 和 bucket hit rate。

## 本地模型 baseline（Qwen3-VL-8B-Instruct）

如果你在自己的服务器上下载了本地模型，可以不走外部 API，直接运行：

### direct

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --mode direct \
  --task-profile structured \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --output artifacts/point1/direct-localqwen-balanced_test_13x5.json
```

### 5-shot

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --mode five_shot \
  --task-profile structured \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --few-shot-parquet train-00001-of-00002.parquet train-00002-of-00002.parquet \
  --few-shot-registry src/benchmark/splits/constructionsite10k_balanced_dev_15x5.json \
  --few-shot-split balanced_dev_15x5 \
  --output artifacts/point1/fiveshot-localqwen-balanced_test_13x5.json
```

如果你想先只看纯分类能力，不输出 bbox：

```bash
--task-profile classification_only
```

更多实现状态见：

- `docs/superpowers/specs/2026-04-10-point1-foundation-design.md`
- `docs/superpowers/plans/2026-04-10-point1-foundation.md`
- `docs/09_point1_foundation_status.md`
- `docs/11_beginner_workflow.md`
- `docs/12_point1_api_baseline.md`
- `docs/13_local_qwen_baseline.md`
