# 13 Local Qwen Baseline

## 目标

给 Point 1 baseline 增加一个**不依赖外部 API** 的本地模型入口，便于在你自己的服务器上直接加载 `Qwen3-VL-8B-Instruct` 运行：

- direct
- 5-shot
- structured
- classification-only

## 推荐环境

建议在服务器上单独准备一个 conda 环境。

### 1. 创建环境

```bash
conda create -n graduation-project python=3.11 -y
conda activate graduation-project
```

### 2. 安装仓库本身

```bash
pip install -e .
```

### 3. 安装本地 Qwen 依赖

根据 Qwen3-VL 与 Transformers 官方文档，本地推理至少需要：

- `torch`
- `accelerate`
- `pillow`
- `transformers`

推荐命令：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install accelerate pillow
pip install git+https://github.com/huggingface/transformers
```

如果你的 CUDA 版本不是 12.4，需要把上面的 PyTorch 源地址换成对应版本。

## 下载模型

你可以把模型下载到任意本地目录，例如：

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir /data/models/Qwen3-VL-8B-Instruct
```

或使用你自己习惯的下载方式。

## 运行 direct baseline

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /home/bml/storage/qwen3_models \
  --mode direct \
  --task-profile structured \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --output artifacts/point1/direct-localqwen-balanced_test_13x5.json
```

## 推荐运行顺序

建议按下面顺序跑：

1. `direct + classification_only`
2. `5-shot + classification_only`
3. `direct + structured`
4. `5-shot + structured`

这样可以先判断模型本身的分类能力，再看 bbox / structured 输出是否拖后腿。

## 四组标准命令

### 1. direct + classification_only

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /home/bml/storage/qwen3_models \
  --mode direct \
  --task-profile classification_only \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --output artifacts/point1/directcls-localqwen-balanced_test_13x5.json
```

### 2. 5-shot + classification_only

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /home/bml/storage/qwen3_models \
  --mode five_shot \
  --task-profile classification_only \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --few-shot-parquet train-00001-of-00002.parquet train-00002-of-00002.parquet \
  --few-shot-registry src/benchmark/splits/constructionsite10k_balanced_dev_15x5.json \
  --few-shot-split balanced_dev_15x5 \
  --output artifacts/point1/fiveshotcls-localqwen-balanced_test_13x5.json
```

### 3. direct + structured

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /home/bml/storage/qwen3_models \
  --mode direct \
  --task-profile structured \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --output artifacts/point1/direct-localqwen-balanced_test_13x5.json
```

### 4. 5-shot + structured

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /home/bml/storage/qwen3_models \
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

## 只跑少量样本做 smoke test

如果你想先快速测试，只要给命令额外加：

```bash
--limit 5
```

例如：

```bash
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /home/bml/storage/qwen3_models \
  --mode direct \
  --task-profile classification_only \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --limit 5 \
  --output artifacts/point1/directcls-localqwen-limit5.json
```

## 输出文件

每次运行会生成：

1. 主结果文件  
   例如：
   - `artifacts/point1/direct-localqwen-balanced_test_13x5.json`

2. summary 文件  
   例如：
   - `artifacts/point1/direct-localqwen-balanced_test_13x5.summary.json`

其中 summary 会记录：

- `num_records`
- `num_success`
- `num_failures`
- `mode`
- `task_profile`

## 跑完后的结果分析

如果你已经跑完一对结果，例如：

- `directcls-localqwen-balanced_test_13x5.json`
- `fiveshotcls-localqwen-balanced_test_13x5.json`

可以直接运行：

```bash
conda activate graduation-project
python scripts/analyze_point1_baselines.py \
  --direct-output artifacts/point1/directcls-localqwen-balanced_test_13x5.json \
  --few-shot-output artifacts/point1/fiveshotcls-localqwen-balanced_test_13x5.json \
  --registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --subset-name balanced_test_13x5 \
  --output artifacts/point1/localqwen-cls-comparison.json
```

如果分析 structured 两组，就把输入文件改成：

- `direct-localqwen-balanced_test_13x5.json`
- `fiveshot-localqwen-balanced_test_13x5.json`

## 作者风格 direct / 5-shot 全测试集

现在脚本额外支持：

- `--prompt-style author_vqa`：改用更接近作者仓库的稀疏 rule-dict VQA 输出格式；
- 不传 `--target-registry` / `--target-split`：直接跑完整 `test.parquet`；
- `--few-shot-example-profile author_train_mimic`：使用 **train** 中固定 5 张示例，
  组合上贴近作者 few-shot（clean、rule1+3、rule1、rule4、rule1+2），
  但避免直接拿 test 图像做 in-context example，保持 benchmark 口径更干净。

### 1. author-style direct（full test）

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /home/bml/storage/qwen3_models \
  --mode direct \
  --prompt-style author_vqa \
  --task-profile structured \
  --target-parquet test.parquet \
  --output artifacts/point1/direct-localqwen-authorvqa-fulltest.json
```

### 2. author-style 5-shot（full test）

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /home/bml/storage/qwen3_models \
  --mode five_shot \
  --prompt-style author_vqa \
  --task-profile structured \
  --target-parquet test.parquet \
  --few-shot-parquet train-00001-of-00002.parquet train-00002-of-00002.parquet \
  --few-shot-example-profile author_train_mimic \
  --output artifacts/point1/fiveshot-localqwen-authorvqa-fulltest.json
```

### 3. 生成 full test 分 rule precision / recall 表

```bash
conda activate graduation-project
python scripts/analyze_point1_baselines.py \
  --direct-output artifacts/point1/direct-localqwen-authorvqa-fulltest.json \
  --few-shot-output artifacts/point1/fiveshot-localqwen-authorvqa-fulltest.json \
  --target-parquet test.parquet \
  --output artifacts/point1/localqwen-authorvqa-fulltest-comparison.json
```

运行后：

- `comparison.json` 会包含 direct / five-shot 两组 summary；
- 每组 summary 里都有 `rule_metrics`；
- 同时终端会直接打印 Markdown 表格，可直接贴到论文或实验记录中。

## 说明

本地模型入口和 API baseline 使用的是同一套：

- 子集 registry
- prompt 逻辑
- JSON parser
- 结果导出格式

因此你后面可以直接把本地模型结果和 API baseline 结果做并排比较。
