# AGENTS.md

## 项目身份
这是一个硕士毕业设计/论文项目，主题为施工场景安全隐患识别。
论文结构固定为：
1. Point 1：ConstructionSite10k 四规则闭域、单图、证据链驱动、可解释违规识别。
2. Point 2：知识增强的 RAG / PEFT(LoRA/QLoRA) 协同方法，面向典型规则违规识别的跨场景鲁棒性提升。
3. Point 3：原型系统与工程闭环验证。

## 最高优先级约束
1. Point 1 与 Point 2 必须在实现层解耦。
2. 代码必须可读、清晰、模块边界明确，优先简单直接的实现。
3. 优先保证 benchmark 口径正确，再优化模型复杂度。
4. 不得为了“更聪明”而牺牲可复现性、可审计性与可测试性。

## 强制架构约束
- `src/point1` 只允许依赖：`src/common`、`src/benchmark`、`src/eval`。
- `src/point1` 禁止依赖：RAG、知识库检索、外部国标检索、LoRA/QLoRA、训练器、微调器、系统服务层。
- `src/point2` 可以依赖：`src/common`、`src/benchmark`、`src/eval`、`src/knowledge`、`src/finetune`。
- `src/point2` 不得直接 import `src/point1` 的算法实现；如果需要复用，只能通过公共契约、公共工具或持久化 artifact。
- Point 3 只消费 Point 1 / Point 2 已稳定的接口，不反向侵入研究代码。

## Point 1 范围（不可越界）
- 仅面向 ConstructionSite10k 内置四条规则。
- 仅处理单张图像。
- 主输出为结构化 JSON，而不是自由文本报告。
- 允许使用局部 crop、candidate generation、predicate extraction、rule executor、显式 explanation slots。
- 不允许引入外部规范检索、RAG、LoRA、QLoRA、额外规则体系、视频主实验。

## Point 2 范围
- Point 2 研究“规则知识增强 + RAG + LoRA/QLoRA / PEFT”的协同作用。
- 允许接入企业制度、国标条款、相似案例检索与参数高效微调。
- Point 2 的目标是跨场景鲁棒性、precision / grounding IoU / 解释一致性的提升。
- Point 2 不得破坏 Point 1 的闭域 benchmark 实验口径。

## Point 1 baseline / BML 运行约定
- Point 1 baseline 当前优先支持 **本地 Qwen** 路径；BML 默认模型路径约定为：
  - `/home/bml/storage/qwen3_models`
- 与 Point 1 baseline 相关的实验，默认先做 smoke test，再做 full test：
  - 先用 `--limit 5` 或 `--limit 20` 验证 parse / output / summary 正常；
  - 确认无误后再跑完整 `test.parquet`。
- 如果目标是补 **per-rule precision / recall / F1** 表，默认优先使用：
  - `--task-profile classification_only`
  - `structured` 主要用于 bbox / grounding 相关分析，不是默认 first pass。
- subset 与 full test 口径必须区分：
  - `balanced_test_13x5`：仅用于 smoke test、prompt 调试、快速对比；
  - `test.parquet`：用于正式 full test 主表与分 rule 指标。
- 当需求包含“全测试集 / full test / 分 rule 表”时，默认应走 full test，不要自动退回 subset。
- 仓库支持 `--prompt-style author_vqa`，用于接近作者仓库的 VQA prompt 契约。
- 作者版本 5-shot 默认采用 **无泄漏版**：
  - direct：使用作者 prompt；
  - five-shot：使用作者 prompt 模板，但 few-shot 示例必须来自 **train-only 固定样本**。
- 官方仓库中的 5-shot 示例来自 `test_split`；默认实现不得把这些 test 示例作为主 benchmark 的 in-context examples。
- 如果以后需要“严格照搬作者 test-shot”的实验，必须作为单独显式 profile / 单独实验存在，不能覆盖默认口径。
- 当前默认 few-shot 示例必须固定，不允许每轮随机抽样；若使用 `author_train_mimic`，应继续保持 train-only 固定 image IDs 策略。

## 代码风格
- 使用 Python `src/` 布局。
- 所有公开函数、类、dataclass 都要写类型标注。
- 单个文件尽量保持职责单一；超过 300 行前先考虑拆分。
- 禁止在 notebook 中沉淀核心逻辑；notebook 只用于探索与可视化。
- 配置放在 `configs/`；禁止把路径、阈值、模型名硬编码在源码里。
- 所有 I/O 契约要用 dataclass / pydantic / TypedDict 之一显式定义。
- 每个模块都要有简洁 docstring，说明输入、输出、假设、限制。

## 测试与检查
每次改动后，尽量保持以下命令可通过：
- `ruff check .`
- `ruff format .`
- `pytest -q`

如果某次任务涉及 schema、评测、输出协议或公共接口，必须：
- 更新对应 `docs/` 文档。
- 增加或更新测试。
- 明确写出变更影响面。

## 与外部仓库的关系
- ConstructionSite10k 官方实现仓库是“benchmark / eval / 数据格式语义参考”，不是新仓库的骨架模板。
- 可以参考其评测脚本、字段命名、示例推理与样例 annotation。
- 不要照搬其目录风格、临时 notebook 结构或特定模型推理代码。

## 与旧仓库的关系
- 旧仓库不是主工作树。
- 如果旧仓库中有必须保留的内容，只允许迁移：
  1. 已冻结的 subset registry
  2. 已确认口径正确的评测桥接逻辑
  3. 具有论文价值的结果表/错误分析素材
- 禁止把旧仓库的历史耦合结构直接带入新仓库。

## 优先阅读顺序
1. `docs/01_project_scope.md`
2. `docs/02_repo_blueprint.md`
3. `docs/03_benchmark_constructionsite10k.md`
4. `docs/04_point1_spec.md`
5. `docs/05_point2_spec.md`
6. `docs/07_build_sequence.md`
7. `code_review.md`
