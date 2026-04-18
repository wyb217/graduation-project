# 07 Build Sequence

## 原则

项目已经不在“纯初始化”阶段，因此当前顺序不应再是盲目从 Phase 0 重来，而应以：

1. 保护已经成立的 Point 1 主线；
2. 收紧文档与实验口径；
3. 再决定后续实现顺序。

## 当前完成度快照

### 已基本完成

- Phase 0：仓库初始文档与研究边界
- Phase 1：ConstructionSite10k loader / parser / split registry
- Phase 2：公共 schema / I/O 契约
- Phase 3：Point 1 black-box baseline、local Qwen baseline、eval bridge

### 已部分完成

- Phase 4：Point 1 主方法
  - Rule 1 主线已推进到 full test 阶段性结果
  - Rule 2 / Rule 3 / Rule 4 仍未达到同等级成熟度

### 尚未开始到稳定阶段

- Phase 5：Point 2 baseline
- Phase 6：Point 2 主方法
- Phase 7：Point 3 scaffold

## 当前推荐顺序

### Step 1：统一文档与运行上下文

- 固定 BML / GitHub / Gitee / 数据路径 / detector 缓存路径
- 让 Point 1 / Point 2 的当前描述与真实进展一致
- 保留结果文件，不把阶段性结果误删

### Step 2：修正 Point 1 black-box baseline 的汇总口径

- 修 parser / summary
- 使用已有 raw outputs 做离线重解析 / 重评分
- 区分“探索性 baseline”与“正式可用表格”

### Step 3：做 Rule 1 full test error anatomy

重点拆解：

- FP
- FN
- unknown

优先看：

- person detection
- hard_hat_visible
- lower_body_covered
- toe_covered

### Step 4：压缩 unknown，同时保护 controllability

- selective padding
- crop 几何优化
- gating 严格度调整

要求：

- 不要为了追 recall 直接牺牲可审计性；
- 不要让 Point 1 重新退回黑箱整图判别。

### Step 5：推进 Rule 4，再回到 Rule 2 / Rule 3

推荐顺序：

1. Rule 4 pair reasoning
2. edge-related 模块
3. Rule 2 / Rule 3

### Step 6：在 Point 1 主线收束后启动 Point 2

Point 2 的合理启动顺序是：

1. 知识单元 schema
2. prompt-only / RAG-only
3. PEFT-only
4. RAG+PEFT
5. robustness matrix

### Step 7：最后再做 Point 3

- 服务接口
- artifact 消费
- 人工复核入口
- 报告导出

## 对 Codex 的执行要求

每个阶段结束时至少产出：

- 可运行命令
- 对应 tests 或验证证据
- 最小文档更新
- 变更影响面说明

额外要求：

- 不要把 quick-test 结果当成 full test 结论；
- 不要在 Point 1 主线还未收束时仓促切向 Point 2；
- 如果是只改文档，也要明确哪些地方属于“当前事实”，哪些地方只是“后续计划”。
