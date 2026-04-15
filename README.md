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

## BML 平台 Git 约定

如果你是在 **BML 平台** 上运行本仓库，请默认采用以下 Git 约定：

- BML 上的仓库是直接从 **gitee clone** 下来的；
- 因此 BML 上的 `origin` 默认就是 **gitee**；
- 在 BML 上拉取或推送更新时，默认使用：

```bash
git fetch origin
git pull origin feature/point1-foundation
```

不要在 BML 上默认把 `origin` 理解成 GitHub。

## BML 平台数据路径约定

如果你是在 **BML 平台** 上运行数据相关脚本，请默认把 ConstructionSite10k 数据根目录设为：

- `/home/bml/storage/constructionsite10k`

推荐先执行：

```bash
export CS10K_ROOT=/home/bml/storage/constructionsite10k
```

下面所有涉及 `train-00001-of-00002.parquet`、`train-00002-of-00002.parquet`、`test.parquet` 的命令，
如果你在 BML 上运行，都优先写成 `${CS10K_ROOT}/...` 的绝对路径。

## BML 平台模型接入约定

如果你是在 **BML 平台** 上推进 Point 1，默认优先使用：

- **本地 Qwen3-VL**

而不是优先占用远端 ModelScope / API 配额。

推荐先执行：

```bash
export QWEN3_VL_ROOT=/home/bml/storage/qwen3_models
```

当前在 BML 上的默认建议是：

1. Rule 1 主方法里的 predicate extraction 优先走 `local_qwen`
2. black-box baseline 优先走 `scripts/run_point1_local_qwen_baseline.py`
3. 只有在需要做远端 provider 对照时，再使用 `modelscope`

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
data_root = Path("/home/bml/storage/constructionsite10k")
dataset = ConstructionSite10kDataset.from_parquet(
    [
        data_root / "train-00001-of-00002.parquet",
        data_root / "train-00002-of-00002.parquet",
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

1. Rule 1 small closed-loop（26 张 clean+rule1）
2. Rule 4 pair reasoning
3. edge-related modules 与 Rule 2/3

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
  "${CS10K_ROOT}/train-00001-of-00002.parquet" \
  "${CS10K_ROOT}/train-00002-of-00002.parquet" \
  --subset-name balanced_dev_15x5 \
  --output src/benchmark/splits/constructionsite10k_balanced_dev_15x5.json
```

如需从 parquet 重新生成 test 子集：

```bash
conda activate graduation-project
python scripts/build_balanced_subset_registry.py \
  "${CS10K_ROOT}/test.parquet" \
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
  --target-parquet "${CS10K_ROOT}/test.parquet" \
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
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --few-shot-parquet "${CS10K_ROOT}/train-00001-of-00002.parquet" "${CS10K_ROOT}/train-00002-of-00002.parquet" \
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

## Rule 1 小闭环运行命令

如果你想先只跑 Rule 1 的最小方法闭环，可以用冻结的 26 张子集：

- `balanced_test_13x5_clean`
- `balanced_test_13x5_rule1`

命令如下：

```bash
conda activate graduation-project
python scripts/run_point1_rule1_pipeline.py \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split-names balanced_test_13x5_clean balanced_test_13x5_rule1 \
  --output artifacts/point1/rule1-smallloop-balanced_test_clean_rule1.json
```

如果你需要做远端 provider 对照，把 Rule 1 的谓词提取改成 OpenAI-compatible VLM（例如 ModelScope），可以额外加：

```bash
conda activate graduation-project
python scripts/run_point1_rule1_pipeline.py \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split-names balanced_test_13x5_clean balanced_test_13x5_rule1 \
  --predicate-backend vlm \
  --provider modelscope \
  --output artifacts/point1/rule1-smallloop-vlm-modelscope-balanced_test_clean_rule1.json
```

如果你在 BML 上已经有 **本地 Qwen3-VL 模型**，也可以直接把 Rule 1 的谓词提取改成 `local_qwen`：

```bash
export QWEN3_VL_ROOT=/home/bml/storage/qwen3_models

python scripts/run_point1_rule1_pipeline.py \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split-names balanced_test_13x5_clean balanced_test_13x5_rule1 \
  --predicate-backend local_qwen \
  --model-path "${QWEN3_VL_ROOT}" \
  --output artifacts/point1/rule1-smallloop-localqwen-balanced_test_clean_rule1.json
```

这条本地版路径的目的就是：

- 不占用远端 API / ModelScope 配额；
- 直接在 BML 上做 `candidate -> local qwen predicate extraction -> executor -> explanation`；
- 保持 Rule 1 executor / explanation 与远端 VLM 版一致，便于做 apples-to-apples 对比。

如果你想开始解决当前最主要的 `unknown(person_detection)` 问题，可以进一步打开实验性的 detector fallback：

```bash
export QWEN3_VL_ROOT=/home/bml/storage/qwen3_models

python scripts/run_point1_rule1_pipeline.py \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split-names balanced_test_13x5_clean balanced_test_13x5_rule1 \
  --candidate-backend hog_then_torchvision \
  --predicate-backend local_qwen \
  --model-path "${QWEN3_VL_ROOT}" \
  --output artifacts/point1/rule1-smallloop-localqwen-hybriddet-balanced_test_clean_rule1.json
```

这条路径的设计意图是：

- 先跑 OpenCV HOG；
- 只有当 HOG 没有找到任何人时，才回退到 torchvision person detector；
- 先用显式开关做 detector 对照，不直接改当前默认路径。

当前这条 VLM 版小闭环仍然保留：

- person candidate generation：OpenCV HOG
- predicate extraction：VLM over person crop
- executor / explanation：显式规则层

因此它验证的是：

`candidate -> VLM predicate extraction -> executor -> explanation`

当前 VLM predicate 版内部额外增加了两个精度 gate：

- `ppe_applicable`：当前候选是否真的是 Rule 1 关心的 **on-foot worker**
- `head_region_visible`：头盔区域是否真的可见，避免“头部不可见却直接判未戴安全帽”

这里的 `ppe_applicable` 是 **candidate-local** 的：

- 它只回答“这个 candidate 本身是不是 Rule 1 要检查的 on-foot worker”
- 它**不**回答“这张图是不是 PPE 场景”
- 因此如果同一个 worker 同时违反 Rule 1 和 Rule 4，Rule 1 依然可以成立
- nearby excavator / edge / pit 不应自动把 Rule 1 设为不适用

executor 会优先使用这两个 gate 压跨规则误报：

- `ppe_applicable = no` -> 不触发 Rule 1 violation
- `ppe_applicable = unknown` -> 输出 `unknown`
- `head_region_visible != yes` 时，`hard_hat_visible = no` 不能直接判 violation

这个脚本会同时生成：

- `artifacts/point1/rule1-smallloop-balanced_test_clean_rule1.json`
- `artifacts/point1/rule1-smallloop-balanced_test_clean_rule1.summary.json`

其中 summary 关注的是 Rule 1 二分类闭环指标：

- `rule1_precision / recall / f1`
- `clean_hit_rate`
- `rule1_hit_rate`
- `unknown_rate_*`

如果你要在 BML 上跑 65 张 `balanced_test_13x5` 的 Rule 1 负例压力测试，可以直接用：

```bash
conda activate graduation-project
python scripts/run_point1_rule1_pipeline.py \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split-names \
    balanced_test_13x5_clean \
    balanced_test_13x5_rule2 \
    balanced_test_13x5_rule3 \
    balanced_test_13x5_rule4 \
    balanced_test_13x5_rule1 \
  --positive-split-name balanced_test_13x5_rule1 \
  --predicate-backend vlm \
  --provider modelscope \
  --output artifacts/point1/rule1-smallloop-vlm-modelscope-balanced_test_13x5.json
```

这组 65 张 summary 会额外给出：

- `negative_hit_rate`
- `fp_by_bucket`
- `bucket_hit_rate`
- `unknown_rate_by_bucket`

如果你想在 BML 上直接跑 **65 张 local Qwen predicate** 版，也可以用：

```bash
export QWEN3_VL_ROOT=/home/bml/storage/qwen3_models

python scripts/run_point1_rule1_pipeline.py \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split-names \
    balanced_test_13x5_clean \
    balanced_test_13x5_rule2 \
    balanced_test_13x5_rule3 \
    balanced_test_13x5_rule4 \
    balanced_test_13x5_rule1 \
  --positive-split-name balanced_test_13x5_rule1 \
  --predicate-backend local_qwen \
  --model-path "${QWEN3_VL_ROOT}" \
  --output artifacts/point1/rule1-smallloop-localqwen-balanced_test_13x5.json
```

如果你想在 BML 上直接跑 **65 张 local Qwen + detector fallback** 版，可以用：

```bash
export QWEN3_VL_ROOT=/home/bml/storage/qwen3_models

python scripts/run_point1_rule1_pipeline.py \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split-names \
    balanced_test_13x5_clean \
    balanced_test_13x5_rule2 \
    balanced_test_13x5_rule3 \
    balanced_test_13x5_rule4 \
    balanced_test_13x5_rule1 \
  --positive-split-name balanced_test_13x5_rule1 \
  --candidate-backend hog_then_torchvision \
  --predicate-backend local_qwen \
  --model-path "${QWEN3_VL_ROOT}" \
  --output artifacts/point1/rule1-smallloop-localqwen-hybriddet-balanced_test_13x5.json
```

如果你想把当前最佳 Rule 1 路径扩到 **full test 3004**，可以直接用：

```bash
export QWEN3_VL_ROOT=/home/bml/storage/qwen3_models

python scripts/run_point1_rule1_pipeline.py \
  --fulltest \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --candidate-backend hog_then_torchvision \
  --predicate-backend local_qwen \
  --model-path "${QWEN3_VL_ROOT}" \
  --candidate-batch-size 1 \
  --predicate-context-mode crop_only \
  --progress-output artifacts/point1/rule1-smallloop-localqwen-hybriddet-fulltest.progress.json \
  --checkpoint-output artifacts/point1/rule1-smallloop-localqwen-hybriddet-fulltest.checkpoint.json \
  --checkpoint-every 100 \
  --failure-output artifacts/point1/rule1-smallloop-localqwen-hybriddet-fulltest.failures.json \
  --output artifacts/point1/rule1-smallloop-localqwen-hybriddet-fulltest.json \
  --summary-output artifacts/point1/rule1-smallloop-localqwen-hybriddet-fulltest.summary.json
```

这里的 `summary.json` 是 Rule 1 专用 fulltest summary，关注：

- `rule1_precision / recall / f1`
- `rule1_tp / fp / fn`
- `positive_support / negative_support`
- `unknown_rate_overall`

新增可选运行期产物：

- `*.progress.json`：heartbeat / 当前进度
- `*.checkpoint.json`：定期落盘的 partial records
- `*.failures.json`：按 Rule 1 真值导出的 FP / FN / unknown / parse-fail 分析入口

新增可选性能/建模参数：

- `--candidate-batch-size`：同一张图上对多个 person crop 做 local Qwen micro-batch
  - 当前默认建议先用 `1`，把它作为稳定研究口径
  - `>1` 仍属于实验性提速选项，建议先在 65 张子集上验证再上 full test
- `--predicate-context-mode crop_only|crop_with_full_image`：
  - `crop_only`：只看 candidate crop
  - `crop_with_full_image`：同时附带整图上下文，但仍保持 candidate-local predicate / executor

如果你还要继续导出 official-style 预测文件，可以接着运行：

```bash
python scripts/run_point1_eval.py \
  --baseline-output artifacts/point1/rule1-smallloop-localqwen-hybriddet-fulltest.json \
  --official-output artifacts/point1/rule1-smallloop-localqwen-hybriddet-fulltest.official.json \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --summary-output artifacts/point1/rule1-smallloop-localqwen-hybriddet-fulltest.eval-summary.json
```

如果你还想导出官方风格预测文件，可以继续运行：

```bash
conda activate graduation-project
python scripts/run_point1_eval.py \
  --baseline-output artifacts/point1/rule1-smallloop-balanced_test_clean_rule1.json \
  --official-output artifacts/point1/rule1-smallloop-balanced_test_clean_rule1.official.json
```

## 本地模型 baseline（Qwen3-VL-8B-Instruct）

如果你在自己的服务器上下载了本地模型，可以不走外部 API，直接运行：

### direct

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --mode direct \
  --task-profile structured \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
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
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --few-shot-parquet "${CS10K_ROOT}/train-00001-of-00002.parquet" "${CS10K_ROOT}/train-00002-of-00002.parquet" \
  --few-shot-registry src/benchmark/splits/constructionsite10k_balanced_dev_15x5.json \
  --few-shot-split balanced_dev_15x5 \
  --output artifacts/point1/fiveshot-localqwen-balanced_test_13x5.json
```

如果你想先只看纯分类能力，不输出 bbox：

```bash
--task-profile classification_only
```

### author-style 全测试集口径

如果你要补作者风格的 direct / 5-shot，并直接在完整 `test.parquet`
上导出分 rule precision / recall 表，可以用：

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /home/bml/storage/qwen3_models \
  --mode direct \
  --prompt-style author_vqa \
  --task-profile structured \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --output artifacts/point1/direct-localqwen-authorvqa-fulltest.json

python scripts/run_point1_local_qwen_baseline.py \
  --model-path /home/bml/storage/qwen3_models \
  --mode five_shot \
  --prompt-style author_vqa \
  --task-profile structured \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --few-shot-parquet "${CS10K_ROOT}/train-00001-of-00002.parquet" "${CS10K_ROOT}/train-00002-of-00002.parquet" \
  --few-shot-example-profile author_train_mimic \
  --output artifacts/point1/fiveshot-localqwen-authorvqa-fulltest.json

python scripts/analyze_point1_baselines.py \
  --direct-output artifacts/point1/direct-localqwen-authorvqa-fulltest.json \
  --few-shot-output artifacts/point1/fiveshot-localqwen-authorvqa-fulltest.json \
  --target-parquet "${CS10K_ROOT}/test.parquet" \
  --output artifacts/point1/localqwen-authorvqa-fulltest-comparison.json
```

说明：

- `author_vqa` 使用更接近作者仓库的 sparse rule-dict 输出格式；
- full test 分析时不需要 `--target-registry` / `--target-split`；
- `author_train_mimic` few-shot 示例固定来自 **train**，避免把 test 图像直接当作
  in-context example，尽量保持 benchmark 口径正确。

## Rule 1 主链路（轻量真实视觉）

仓库现在包含 Point 1 的第一条方法链路：

- `point1.candidates`：person candidate generation
- `point1.predicates.rule1`：Rule 1 PPE predicates
- `point1.executor.rule1`：Rule 1 decision executor
- `point1.pipelines.rule1`：`candidate -> predicate -> executor -> explanation`

当前实现使用：

- OpenCV HOG people detector 生成人框
- 基于 crop 的轻量 PPE heuristic 判别

当前 `pipeline` 的工作顺序是：

1. 为单张图像生成人候选；
2. 对每个候选抽取 Rule 1 predicates；
3. 由显式 executor 给出 `violation / no_violation / unknown`；
4. 把 executor 的结构化结果映射成 `reason_slots` 与 `reason_text`。

这里的 `explain` 不是自由文本生成，而是 **executor 驱动的结构化解释**：

- `reason_slots` 来自缺失项 / unknown 项 / 候选主体；
- `reason_text` 由模板函数从这些结构化状态生成；
- 因此 explanation 与 predicates / executor 保持一一对应。

这条链路当前主要用于：

- 主方法骨架搭建
- smoke test
- Rule 1 单规则方法验证

当前 Point 1 中 **VLM 的角色仍主要在 baseline**：

- direct / 5-shot / author-style prompt 这些对照组已经使用 VLM；
- 当前 Rule 1 主链路本身还没有把 VLM 放进 predicate extraction。

后续更推荐的演化方向不是“统一 VLM 最终判别”，而是：

- `candidate -> unified VLM predicate judge -> executor -> explanation`

也就是说，让 VLM 统一做局部谓词判别，而把最终规则决策继续保留在 executor 中。

如果你要在新环境里运行 Rule 1 真实视觉链路，确保已经安装：

```bash
pip install -e .
```

更多实现状态见：

- `docs/superpowers/specs/2026-04-10-point1-foundation-design.md`
- `docs/superpowers/plans/2026-04-10-point1-foundation.md`
- `docs/09_point1_foundation_status.md`
- `docs/11_beginner_workflow.md`
- `docs/12_point1_api_baseline.md`
- `docs/13_local_qwen_baseline.md`
