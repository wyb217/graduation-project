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
  --model-path /data/models/Qwen3-VL-8B-Instruct \
  --mode direct \
  --task-profile structured \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --output artifacts/point1/direct-localqwen-balanced_test_13x5.json
```

## 运行 5-shot baseline

```bash
conda activate graduation-project
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /data/models/Qwen3-VL-8B-Instruct \
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

## 只做分类，不输出 bbox

如果你想先看纯规则识别，而不让 bbox 影响解析稳定性：

```bash
--task-profile classification_only
```

例如：

```bash
python scripts/run_point1_local_qwen_baseline.py \
  --model-path /data/models/Qwen3-VL-8B-Instruct \
  --mode direct \
  --task-profile classification_only \
  --target-parquet test.parquet \
  --target-registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --target-split balanced_test_13x5 \
  --output artifacts/point1/directcls-localqwen-balanced_test_13x5.json
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

## 说明

本地模型入口和 API baseline 使用的是同一套：

- 子集 registry
- prompt 逻辑
- JSON parser
- 结果导出格式

因此你后面可以直接把本地模型结果和 API baseline 结果做并排比较。
