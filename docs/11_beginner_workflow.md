# 11 Beginner Workflow

## 目标

这个项目后续会尽量保持：

- 环境简单；
- 命令少；
- 文件职责清楚；
- 代码容易读。

因此默认工作流改为 **conda-first**，并提供尽量简单的脚本入口。

## 环境准备

第一次进入项目时，推荐这样做：

```bash
conda env create -f environment.yml
conda activate graduation-project
```

如果以后依赖发生变化：

```bash
conda env update -f environment.yml --prune
```

## 你最常用的命令

### 一条命令做完整检查

```bash
python scripts/dev.py all
```

它会自动执行：

1. `ruff format .`
2. `ruff check .`
3. `pytest -q`

### 单独执行某一步

```bash
python scripts/dev.py format
python scripts/dev.py check
python scripts/dev.py test
```

## 怎么理解这些命令

如果你暂时不熟悉工程化工具，可以这样理解：

- `format`：把代码排整齐
- `check`：找明显问题
- `test`：确认功能没坏

这套流程主要是为了帮助我在持续改代码时，不把项目越改越乱。

## 后续代码风格约定

后面整个项目会尽量遵守：

1. 一个文件只做一件事；
2. 一个函数尽量只做一个明确动作；
3. 命名尽量直白，不堆缩写；
4. 尽量不引入花哨框架；
5. 能写简单版，就不写复杂版。

## BML 数据路径约定

如果你是在 BML 上跑数据相关脚本，先执行：

```bash
export CS10K_ROOT=/home/bml/storage/constructionsite10k
```

后面的 parquet 示例都默认从这个目录读取。

## BML 模型接入约定

如果你是在 BML 上推进 Point 1，默认先执行：

```bash
export QWEN3_VL_ROOT=/home/bml/storage/qwen3_models
```

并优先采用：

- 本地 Qwen baseline
- Rule 1 的 `local_qwen` predicate backend

尽量不要把 BML 端的日常实验默认建立在远端 ModelScope 配额上。

## 你后续最常见的两种操作

### 1. 做完改动后检查项目

```bash
conda activate graduation-project
python scripts/dev.py all
```

### 2. 重新生成快速测试子集

```bash
conda activate graduation-project
python scripts/build_balanced_subset_registry.py \
  "${CS10K_ROOT}/train-00001-of-00002.parquet" \
  "${CS10K_ROOT}/train-00002-of-00002.parquet" \
  --subset-name balanced_dev_15x5 \
  --output src/benchmark/splits/constructionsite10k_balanced_dev_15x5.json
```
