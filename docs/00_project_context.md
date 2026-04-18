# 00 Project Context

## 文档作用

这份文档是当前仓库的统一锚点，用来固定三类事实：

1. 当前项目到底进行到哪里；
2. 哪些运行配置是长期必须保留的；
3. 哪些文档是“当前有效”，哪些更偏历史蓝图或参考资料。

当不同文档之间出现时间差时，优先以本文件、`docs/04_point1_spec.md`、`docs/05_point2_spec.md` 与 `POINT1_BASELINE_RESULTS.md` 为准。

## 当前项目状态

- 根目录现在不应再被理解成单纯的 “from-scratch handoff pack”。
- 当前真实的可运行实现位于 `.worktrees/feature-point1-foundation`。
- Point 1 已经从 benchmark / baseline 阶段推进到 Rule 1 主方法的 full test 阶段性结果。
- Point 2 已经完成问题收敛，但还没有稳定实现与正式实验结果。
- `POINT1_BASELINE_RESULTS.md` 与 `CONSTRUCTIONSITE10K_DATASET_STATS.md` 属于需要长期保留的结果文件。

## 必须保留的运行事实

### 1. 平台分工

- 本地开发机：主要负责代码编辑、文档整理、轻量检查和结果汇总。
- BML：默认重实验平台，承担本地模型推理、长时间运行和大部分正式实验命令。

### 2. Git 远端与同步约定

当前本地仓库远端为：

- `origin = git@github.com:wyb217/graduation-project.git`
- `gitee = git@gitee.com:wyb7927/graduation_proj.git`

默认同步规则：

- 只要是准备保留的仓库更新，默认同时推送 GitHub 与 Gitee。
- 本地常用命令是：

```bash
git push origin <branch>
git push gitee <branch>
```

BML 特殊约定：

- BML 上的仓库可能直接由 Gitee clone 下来；
- 因此 BML 上的 `origin` 可能已经是 Gitee；
- 在 BML 上执行 pull / push 之前，先运行：

```bash
git remote -v
```

- 不要先验地把 BML 上的 `origin` 理解成 GitHub。

### 3. 数据、模型与缓存路径

默认运行路径约定如下：

```bash
export CS10K_ROOT=/home/bml/storage/constructionsite10k
export QWEN3_VL_ROOT=/home/bml/storage/qwen3_models
export TORCH_HOME=/home/bml/storage/torch_cache
mkdir -p "${TORCH_HOME}/hub/checkpoints"
```

说明：

- `CS10K_ROOT`：ConstructionSite10k 的默认数据根目录。
- `QWEN3_VL_ROOT`：BML 本地 Qwen 模型目录。
- `TORCH_HOME`：PyTorch / torchvision 权重缓存目录。

当前 torchvision fallback detector 的权重缓存位置为：

```text
${TORCH_HOME}/hub/checkpoints
```

当前使用的默认 COCO person detector 权重文件为：

```text
fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
```

如需手动预下载，可使用：

```bash
wget -c https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth \
  -O "${TORCH_HOME}/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
```

### 4. 当前关键实现位置

- 活跃实现工作树：`.worktrees/feature-point1-foundation`
- Rule 1 detector 代码：`.worktrees/feature-point1-foundation/src/point1/candidates/person.py`
- 本地 provider 配置模板：`.worktrees/feature-point1-foundation/configs/system/providers.example.json`
- 本地私有 provider 配置：`.worktrees/feature-point1-foundation/configs/system/providers.local.json`

### 5. 当前 Point 1 默认运行口径

当前最可信、最值得继续推进的 Rule 1 主线是：

```text
hog_then_torchvision -> local_qwen predicate -> executor -> explanation
```

默认研究口径：

- 先跑 smoke test，再跑 full test；
- `balanced_test_13x5` 只用于快速比较与调试；
- `test.parquet` 才用于正式 full test 结论；
- `classification_only` 更适合先看 black-box 分类能力；
- `structured` 更偏向 bbox / grounding / 协议稳定性分析。

## 当前文档导航

### 第一优先级

1. `README.md`
2. `docs/00_project_context.md`
3. `docs/01_project_scope.md`
4. `docs/04_point1_spec.md`
5. `docs/05_point2_spec.md`
6. `POINT1_BASELINE_RESULTS.md`
7. `CONSTRUCTIONSITE10K_DATASET_STATS.md`
8. `docs/07_build_sequence.md`

### 第二优先级

- `docs/02_repo_blueprint.md`：仓库架构蓝图
- `docs/03_benchmark_constructionsite10k.md`：benchmark 参考
- `docs/06_external_refs.md`：外部资料入口
- `docs/08_first_codex_prompt.md`：早期开仓提示词
- `.worktrees/feature-point1-foundation/docs/09_*` 到 `17_*`：更细的阶段性进展、审计与总结

## 结果文件保留策略

根目录下以下文件默认不删除：

- `POINT1_BASELINE_RESULTS.md`
- `CONSTRUCTIONSITE10K_DATASET_STATS.md`

worktree 中以下目录也视为需要保留的结果区：

- `.worktrees/feature-point1-foundation/artifacts/point1/`

说明：

- `*.progress.json`、`*.checkpoint.json`、`*.failures.json` 主要是运行与排错材料；
- `*.summary.json`、`*.official.json`、`*.eval-summary.json` 更接近对外汇总结果；
- 这些文件应整理、分级使用，但不应在没有确认价值之前随意删除。
