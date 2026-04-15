# 15 Point 1 Progress Log

## 文档目的

这份文档用于持续记录 Point 1 的详细进展，重点回答：

1. 已经做了哪些事情；
2. 当前拿到了什么结果；
3. 当前主线是什么；
4. 当前哪里有问题；
5. 下一步应该先做什么。

和 `docs/09_point1_foundation_status.md` 的关系是：

- `docs/09`：阶段概览，适合快速对齐状态；
- `docs/15`：详细实验记录，适合后续回顾、排错和论文整理。

---

## 当前主线

当前 Point 1 最重要的研究主线是：

`hog_then_torchvision -> local_qwen predicate -> executor -> explanation`

它对应的目标不是“整图黑箱判断”，而是：

- 用显式 candidate 驱动裁剪；
- 用 local Qwen 做局部谓词提取；
- 用 executor 做可审计的规则决策；
- 用 explanation 暴露结构化原因。

当前默认稳定运行口径是：

- `--candidate-backend hog_then_torchvision`
- `--predicate-backend local_qwen`
- `--candidate-batch-size 1`
- `--predicate-context-mode crop_only`
- `--crop-padding-profile none`

当前 BML 侧推荐同时固定：

- `export CS10K_ROOT=/home/bml/storage/constructionsite10k`
- `export QWEN3_VL_ROOT=/home/bml/storage/qwen3_models`
- `export TORCH_HOME=/home/bml/storage/torch_cache`

其中 `TORCH_HOME` 的目的不是影响模型逻辑，而是把 PyTorch / torchvision 的权重缓存固定到持久化目录，
避免 `hog_then_torchvision` 在新容器或新会话里重复下载 detector 权重。

---

## 已完成事项

### 1. Point 1 基础设施

已完成：

- typed schema
- parquet / JSON loader
- split registry
- benchmark 子集冻结
- Point 1 结构化输出契约
- API baseline / local Qwen baseline 入口

### 2. Rule 1 主方法链路

已完成：

- person candidate generation
- predicate extraction
- symbolic executor
- explanation mapping
- image-level aggregation

### 3. detector fallback

已从：

- `HOG only`

推进到：

- `hog_then_torchvision`

这一步带来了 Rule 1 recall 的明显提升，说明此前的核心瓶颈之一确实是 `person_detection`。

### 4. full test 运行支持

已完成：

- Rule 1 full test runner
- progress / heartbeat
- checkpoint partial records
- failure export
- BML 侧 `TORCH_HOME` 持久缓存约定

这意味着长时间运行不再完全黑箱，实验可操作性更强。

### 5. Rule 1 命令面简化

已完成：

- `--target-preset balanced65|fulltest`
- `--run-name <name>`

目的不是改变功能，而是：

- 让 BML 上常见运行命令更短；
- 降低手写超长文件名和 split 名的出错概率；
- 保留完整底层参数，便于后续研究型调参。

### 6. crop-only prompt 回归修复

已经完成：

- 恢复 `crop_only` 到旧版 prompt 语义
- 不再在 `crop_only` 中偷偷附加 `Candidate bbox`
- `bbox + full image` 只保留在 `crop_with_full_image` 实验模式下

这样做的原因是：

- `crop_only` 必须是稳定的基线；
- 否则所有后续 A/B 都会被 prompt 漂移污染。

---

## 当前关键结果

### 1. Rule 1 full test（3004）

当前主线 full test 结果：

- precision: `0.633`
- recall: `0.549`
- F1: `0.588`
- TP / FP / FN: `178 / 103 / 146`
- unknown rate: `0.572`

当前解释：

- precision 明显优于 black-box classification-only baseline；
- 但 unknown 仍然偏高；
- 说明系统已经具备 controllability，但 coverage 还不够。

### 2. 65 张有效旧基线

此前有效的 `balanced_test_13x5` 结果：

- precision: `0.917`
- recall: `0.846`
- F1: `0.880`
- TP / FP / FN: `11 / 1 / 2`

这组结果仍然是当前最关键的 quick-test 参照物。

### 3. batch 路径暴露的问题

在尝试 `candidate_batch_size > 1` 后，曾出现：

- `parse_fail`
- processor padding 相关错误

这说明 batching 仍然是实验性优化，不宜作为默认研究口径。

### 4. batchsize=1 仍然发生的回归

即使把 batch size 恢复到 1，仍观察到一轮 65 张结果退化：

- recall 下滑
- F1 下滑
- FP/FN 同时变坏

当前最可能原因是：

- 在那一版代码里，`crop_only` prompt 已被意外修改；
- candidate gating（尤其 `person_visible / ppe_applicable`）可能被过度收紧。

因此后来已完成 `crop_only` prompt 回归修复。

---

## 当前已知问题

### 1. unknown 率仍然偏高

这对安全审计是有意义的，因为系统更保守；
但对 benchmark 二值对比不理想，因为：

- recall 会被压低；
- precision 里会混入 abstention 带来的保守收益。

### 2. unknown 主要集中在少数谓词

当前最值得优先处理的 unknown 来源仍然是：

- `hard_hat_visible`
- `lower_body_covered`
- `toe_covered`
- `person_detection`

这说明当前最值得先改的不是大模型或复杂系统结构，而是：

- candidate crop 的几何质量；
- gating 的严格程度；
- unknown 的传播策略。

### 3. batching 还不是稳定加速路径

当前 batching 的工程接口已经有了，但研究上还没有证明：

- 它在真实 BML 环境里稳定；
- 它不会改变结果分布；
- 它能带来足够显著的吞吐收益。

因此短期内它只应作为显式实验开关，而不是默认主线。

---

## 当前新增加的实验线：crop padding

为了优先压 `hard_hat_visible / lower_body_covered / toe_covered` 的 unknown，
当前新增一条**默认关闭**的实验线：

- `--crop-padding-profile none|rule1_ppe`

约束：

- 只作用于 `local_qwen` 谓词提取输入；
- 不改 detector 原始候选框；
- 不改输出 bbox；
- 不改 executor 协议。

`rule1_ppe` 的第一版固定扩张比例：

- left/right: `+8%`
- top: `+12%`
- bottom: `+20%`

设计目的：

- 给头部更多上下文，帮助 `hard_hat_visible`
- 给下半身和脚部更多上下文，帮助 `lower_body_covered / toe_covered`

当前它还是实验线，不是默认主线。

---

## 当前建议的实验顺序

### 先做

1. 等待 BML 上“恢复旧 prompt”结果回来
2. 确认 `crop_only + batchsize=1 + none` 是否回到旧有效 65 张基线附近

### 再做

3. 在 `balanced_test_13x5` 上测试：
   - `crop_padding_profile=none`
   - `crop_padding_profile=rule1_ppe`
4. 重点比较：
   - unknown rate
   - precision / recall / F1
   - FP/FN/unknown failure export

### 暂缓

5. 不急着重新推进 batching
6. 不急着把 full-image context 和 crop padding 混在一起

原则是：

- 一次只改一个主要变量；
- 先把 unknown 压下来，再谈进一步复杂化。

---

## 当前结论

截至当前，Point 1 的判断可以概括为：

1. Rule 1 主线已经成立，不再只是原型；
2. detector fallback 已证明有效；
3. current bottleneck 已从“能不能跑起来”转向“如何压 unknown 并保持 controllability”；
4. crop padding 是当前最值得优先尝试的低风险改动之一；
5. batching 目前仍然不是研究主问题。
