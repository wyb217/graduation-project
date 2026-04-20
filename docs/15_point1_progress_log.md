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
- `docs/17`：阶段总结，适合先判断“当前到底能不能算有阶段性结果”；
- `docs/15`：详细实验记录，适合后续回顾、排错和论文整理。

另外，针对“当前已跑结果在 parser / summary 修复后是否还能继续使用”的问题，当前单独维护：

- `docs/16_point1_result_validity_audit.md`

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
- `--max-new-tokens 500`

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

现在 runner 侧还新增了**按图耗时画像**，会随 progress / checkpoint / summary 一起落盘。当前重点跟踪：

- `candidate_ms`
- `predicate_ms`
- `executor_ms`
- `total_ms`
- `candidate_count`
- `fallback_used`
- `predicate_backend`
- `candidate_batch_size`
- `max_new_tokens`

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

### 7. local Qwen 默认吞吐优化

当前默认主线已经补上两项低风险吞吐优化：

- 送入 local Qwen 的 candidate crop 会在不改 bbox 的前提下做最长边 `640px` 的压缩

这里的边界要明确：

- 不改 detector 原始输出 bbox；
- 不改最终结构化输出 bbox；
- 不改 prompt 语义；
- 只降低 local Qwen 谓词提取阶段的推理成本。

当前默认主线仍保持更保守的生成上限：

- `max_new_tokens = 500`

原因是：

- 更短的 generation cap 可以作为后续吞吐实验变量；
- 但在没有通过稳定 canary 验证之前，不应直接升为默认主线配置。

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

### 1.1 Rule 1 crop padding full test（3004）

`crop_padding_profile=rule1_ppe` 的 full test 结果现已补齐：

- precision: `0.652`
- recall: `0.531`
- F1: `0.585`
- TP / FP / FN: `172 / 92 / 152`
- unknown rate: `0.556`

相对默认主线 `crop_padding_profile=none`：

- precision：`0.633 -> 0.652`
- recall：`0.549 -> 0.531`
- F1：`0.588 -> 0.585`
- FP：`103 -> 92`
- FN：`146 -> 152`
- unknown：`1718 -> 1671`

当前判断：

- crop padding 的确能压一部分局部可见性相关 unknown；
- 也能进一步降低 FP；
- 但它没有带来更好的 full test F1；
- 因此当前应把它视为 **precision-oriented / unknown-reduction variant**，
  而不是默认主线替代品。

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

当前代码已经给 `complete_batch()` 增加了更保守的稳定化处理：

- 如果 batch 路径触发 processor / padding 相关异常，
- 会回退到按 candidate 顺序逐条 completion，
- 以保证实验 lane 至少不因实现脆弱性直接中断。

但这仍然不意味着 batching 已经成为默认推荐路径。当前结论仍然保持：

- 默认主线 = `candidate_batch_size=1`
- `candidate_batch_size=2` 及以上只用于后续吞吐实验 lane

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

现在 full test 已经证明：

- 它确实能压 `toe_covered / lower_body_covered / hard_hat_visible` 一部分 unknown；
- 但对 `person_detection` 没有帮助；
- 在当前固定扩张比例下，precision 提升不足以抵消 recall 回落。

因此截至目前：

- `rule1_ppe` 值得保留；
- 但默认 full test 主线仍应保持 `crop_padding_profile=none`。

---

## 当前建议的实验顺序

### 先做

1. 基于 `rule1_ppe` full test failure export，拆第一轮：
   - `unknown -> no_violation`
   - `unknown -> violation`
   - `violation -> unknown/no_violation`
2. 确认是否值得做 selective padding，而不是 always-on padding

### 再做

3. 尝试只在出现 `toe_covered / lower_body_covered / hard_hat_visible` unknown 时触发二次 padding 重判
4. 重点比较：
   - 是否能保住默认主线 recall
   - 是否还能继续压 FP / unknown
   - full test 下的净 F1 变化

### 暂缓

5. 不急着重新推进 batching
6. 不急着把 full-image context 和 crop padding 混在一起
7. 不急着把 detector-guided VLM baseline 升成主线
8. 当前不把 candidate cap 直接升为默认主线

但这里有一条值得明确记录的**后续对照实验**：

- 在现有 image-only baseline 与 Point 1 主方法之间，
- 后续补做一条 **detector-guided VLM baseline**

具体含义是：

- 先用 detector 检出候选目标
- 再把 `label + bbox` 作为显式先验放进提示词
- 最后仍由 VLM 完成判定

这样做的目的不是替代当前主线，而是回答：

> detector 先验本身能给纯 VLM 判定带来多少提升？

当前更推荐把它放在：

- Rule 1
- Rule 4

上先做小规模对照，因为这两类规则最容易体现“显式对象先验”对判定的作用。

目前可参考的近邻论文是：

- [Integration of Object Detection and Small VLMs for Construction Safety Hazard Identification](https://arxiv.org/abs/2604.05210)

这篇论文的相关启发点是：

- 使用 object detector 先定位 worker / machinery
- 再把检测结果嵌入结构化 prompt
- 用 detection-guided VLM 做 construction hazard reasoning

对本仓库而言，这条实验线更适合定位为：

- detector-guided baseline / ablation
- 而不是 Point 1 主方法主线

原因是：

- 它本质上还是让 VLM 做最终判定；
- 而 Point 1 主线的核心价值仍然是：
  - `candidate -> predicate -> executor -> explanation`
  - 即显式证据链而非 detector-conditioned black-box judgement

另外，围绕当前 Rule 1 长尾吞吐问题，仓库已补上一条**默认关闭**的 candidate cap 实验 lane：

- 参数：`--max-candidates-per-image`
- 作用范围：只针对 Rule 1 的 `person candidate`
- 作用位置：在 `Rule1Pipeline.run(...)` 中，candidate 生成之后、谓词提取之前
- 第一版策略：按 detector score 做 top-K 截断

当前定位：

- 只作为 canary lane
- 不进入默认主线
- 初始实验组固定为：
  - `no-cap`
  - `top-6`
  - `top-4`

这样做的原因是：

- 当前 canary1000 已经证明 `candidate_count` 与 `predicate_ms` 近似线性相关；
- 因此 candidate 数控制本身有望同时改善：
  - 吞吐
  - 误报
  - unknown 长尾

但当前 detector 现实也要写清楚：

- `hog_then_torchvision` 只会在 **HOG 零候选** 时才触发 torchvision fallback；
- 因此它能修复的是 **漏检**，而不是 **错检**；
- 如果 HOG 已经给出错误候选，当前链路不会自动再调用 torchvision 进行纠正。

当前对 candidate cap 的定位也应保持清楚：

- 它首先是一个**吞吐实验开关**
- 不是 detector 改造
- 不是新的 Point 1 主方法
- 它回答的是：
  - 在不改变 prompt / executor 的前提下，限制每图进入 VLM 的候选数，能否降低 local Qwen 的长尾耗时

原则是：

- 一次只改一个主要变量；
- 先把 unknown 压下来，再谈进一步复杂化。

---

## 当前结论

截至当前，Point 1 的判断可以概括为：

1. Rule 1 主线已经成立，不再只是原型；
2. detector fallback 已证明有效；
3. current bottleneck 已从“能不能跑起来”转向“如何压 unknown 并保持 controllability”；
4. crop padding full test 已完成，当前更像一个保守变体，而不是默认主线升级；
5. batching 目前仍然不是研究主问题。
