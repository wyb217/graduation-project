# graduation-project

施工场景安全隐患识别硕士毕业设计仓库。

当前最重要的事实不是“从零搭框架”，而是：
- Point 1 已从 baseline 推进到 Rule 1 的 full test 阶段性结果。
- Point 2 已收紧为“知识源变化 / 规则表述变化”下的知识增强 RAG + PEFT 协同研究，不再泛泛表述为未操作化的“跨场景鲁棒性”。
- 重型实验默认在 BML 上运行；当前可运行实现位于 `.worktrees/feature-point1-foundation`。
- 结果文件需要保留，尤其是 `POINT1_BASELINE_RESULTS.md` 与 `CONSTRUCTIONSITE10K_DATASET_STATS.md`。

## 先读这些文档
1. `docs/00_project_context.md`
2. `docs/01_project_scope.md`
3. `docs/04_point1_spec.md`
4. `docs/05_point2_spec.md`
5. `POINT1_BASELINE_RESULTS.md`
6. `docs/07_build_sequence.md`

## 三个研究点
- Point 1：ConstructionSite10k 四规则闭域、单图、证据链驱动、可解释违规识别。
- Point 2：面向知识源变化与规则表述变化的知识增强 RAG / PEFT 协同方法。
- Point 3：消费 Point 1 / Point 2 稳定接口的原型系统与工程闭环验证。

## 当前阶段
- Point 1：Rule 1 主线 `hog_then_torchvision -> local_qwen predicate -> executor -> explanation` 已有 full test 阶段性结果。
- Point 2：完成了问题收敛与边界重写，但尚无稳定实现与正式结果。
- Point 3：仍是接口消费层，不反向侵入研究代码。

## 运行与同步约定
- BML 是默认重实验平台；ConstructionSite10k 默认根目录为 `/home/bml/storage/constructionsite10k`。
- 本地 Qwen 默认目录为 `/home/bml/storage/qwen3_models`，PyTorch / torchvision 缓存默认目录为 `/home/bml/storage/torch_cache`。
- 仓库更新后默认同时推送 GitHub 与 Gitee；具体远端说明、命令和 BML 特殊约定见 `docs/00_project_context.md`。

## 说明
根目录主要负责文档整合、研究边界和结果沉淀；实际可运行实现、脚本、配置与测试目前集中在 `.worktrees/feature-point1-foundation`。
