# 14 Opening Report Summary

> 历史说明：本文档是 2026-04-12 的开题汇报快照，保留其当时的叙述用于追溯。
> 当前 Point 2 的有效定义已经更新，后续应以 `docs/05_point2_spec.md` 为准，而不是直接沿用本文中的旧表述。

## 文档目的

本文档用于开题阶段的简要汇报，回答四个核心问题：

1. 研究点一做什么；
2. 研究点二做什么；
3. 两个研究点为什么要这样拆分；
4. 研究点一目前已经完成了哪些 baseline，结果说明了什么。

阅读提示：

- 本文档是**开题阶段**的汇报快照；
- 其中多数结果仍以 `balance65` quick-test 为主；
- 如果你需要看**当前最新阶段结论**，请优先转到：
  - `docs/17_point1_stage_summary.md`
  - `POINT1_BASELINE_RESULTS.md`

需要强调的是：

- 当前展示的 Point 1 结果主要来自 `balance65` quick-test。
- `balance65` 对应仓库中的 `balanced_test_13x5`。
- 它是一个**快速验证子集**，不是最终论文主表的 full test 结果。

---

## 总体研究结构

本课题采用“两个研究点 + 一个系统点”的结构：

- Point 1：ConstructionSite10k 四规则闭域、单图、证据链驱动、可解释违规识别。
- Point 2：知识增强的 RAG / PEFT 协同方法，面向典型规则违规识别的跨场景鲁棒性提升。
- Point 3：原型系统与工程闭环验证。

其中，Point 1 和 Point 2 必须在实现层明确解耦：

- Point 1 只讨论闭域 benchmark 内的四规则识别。
- Point 2 才引入外部知识、RAG、LoRA / QLoRA / PEFT。

这样拆分的原因是：

- 先把闭域 benchmark 口径跑通，得到可审计、可复现的基础能力基线；
- 再研究知识增强和参数高效微调对跨场景泛化的增益；
- 避免一开始把“规则识别能力”与“知识增强能力”混在一起，导致实验无法归因。

---

## Point 1

### 1. 做什么

Point 1 的目标是：

> 面向 ConstructionSite10k 四规则闭域，完成单张图像下的可解释违规识别。

它不是自由文本问答，而是一个受控的结构化识别任务。输入是一张施工现场图像，输出是每条规则是否违规、违规位置以及简短解释。

当前只面向数据集内置四条规则：

1. Basic PPE
2. Safety harness when working at height without protection
3. Edge protection / warning for underground projects
4. Workers inside excavator blind spots or operating radius

### 2. 为什么做

Point 1 的研究价值在于：

- 先建立一个**闭域、可控、可复现**的 benchmark 问题；
- 将“施工安全隐患识别”收敛成四条明确规则，避免任务边界过宽；
- 为后续 Point 2 提供稳定的比较基线；
- 为 Point 3 系统实现提供稳定的对外接口。

如果 Point 1 不先独立成立，后续加入外部知识、RAG 或 LoRA 后，实验结果很难判断到底是：

- 模型本身更强；
- prompt 更合适；
- 外部知识带来了增益；
- 还是 benchmark 口径本身没有冻结。

### 3. 怎么做

Point 1 的总体技术路线分两层：

#### 第一层：黑箱 baseline

先用现成 VLM 建立最小可运行对照组，用于回答：

- 多模态模型能否稳定接收 ConstructionSite10k 图像；
- 结构化 JSON 是否能稳定输出；
- few-shot 是否能改善协议稳定性；
- 在不引入我们方法的情况下，黑箱模型的闭域能力大致处于什么水平。

当前已经具备的 baseline 维度包括：

- `direct` / `five_shot`
- `structured` / `classification_only`
- API 模型（ModelScope）
- BML 平台本地 Qwen3-VL

#### 第二层：Point 1 主方法

后续会把“整图问答式识别”改造成：

`candidate -> evidence -> predicate -> executor -> explanation`

核心思路是把违规识别拆成可解释链路，而不是仅依赖黑箱回答。

主方法的预期顺序为：

1. Rule 1 真实视觉路径
2. Rule 4 pair reasoning
3. edge module
4. Rule 2 / Rule 3
5. executor + explanation hardening

也就是说，Point 1 的 baseline 不是终点，而是后续证据链方法的比较起点。

---

## Point 2

### 1. 做什么

Point 2 的目标是：

> 面向施工典型规则违规识别，引入知识增强、RAG 与 PEFT，提升跨场景鲁棒性。

Point 2 不再局限于 ConstructionSite10k 的闭域四规则，而是研究：

- 如何接入企业制度、规范条款和相似案例；
- 如何提升跨场景、小样本条件下的 precision、grounding IoU 与 explanation consistency；
- RAG-only、LoRA-only、LoRA+RAG 三者各自的收益边界。

### 2. 为什么做

Point 1 的闭域基线有两个天然上限：

- 只能依赖模型参数中已有的隐式知识；
- 对新场景、新表述、新规则细节的泛化能力有限。

施工安全场景本身又具有明显知识密集特征：

- 规则来自条文，而不是纯视觉常识；
- 很多判断需要“场景 + 规范 + 语义条件”联合成立；
- 仅靠黑箱 VLM 往往容易高 recall、低 precision，出现“过报”。

因此 Point 2 的意义在于：

- 把规则知识显式接入模型；
- 用 retrieval trace 提高可追溯性；
- 用 PEFT 提升特定场景和规则的适应能力；
- 研究知识增强与参数微调是否互补，而不是只做“换更大模型”。

### 3. 怎么做

Point 2 的建议路线是三层协同：

#### 知识层

- 规则条款模板化
- 企业制度结构化
- 相似案例索引
- 检索模块与知识版本管理

#### 模型层

- LoRA / QLoRA / PEFT
- 面向规则识别的数据构造与训练集组织

#### 推理层

- 检索增强输入
- 模板化输出
- 规则一致性约束
- retrieval trace / matched clause 暴露

推荐实验对照为：

- zero-shot
- prompt / CoT
- RAG-only
- LoRA-only
- LoRA+RAG

目前 Point 2 还没有进入稳定结果阶段，当前工作的重点仍然是先把 Point 1 baseline 口径冻结清楚。

---

## Point 1 当前结果

## 1. 当前使用的数据口径

当前已完成结果主要基于 `balance65` quick-test：

- 对应仓库名：`balanced_test_13x5`
- 总数：65 张
- bucket 构成：
  - `clean`: 13
  - `rule1`: 13
  - `rule2`: 13
  - `rule3`: 13
  - `rule4`: 13

它的作用是：

- 用于 prompt 调试；
- 用于 baseline 是否跑通的快速比较；
- 用于先观察 direct / five-shot / structured / classification_only 的相对变化。

它**不是** full test 3004 的正式论文口径，因此不应用于直接替代最终主表。

## 2. 当前较重要的 baseline 结果

### ModelScope API

#### direct + structured

- parse success rate: `4 / 65 = 6.15%`
- macro-F1: `0.033`

结论：

- 直接要求模型输出完整结构化 JSON 时，协议稳定性很差；
- 当前主要问题不是完全不会判断，而是输出格式、bbox 和多字段 JSON 不稳定。

#### 5-shot + structured

- parse success rate: `65 / 65 = 100%`
- macro-F1: `0.680`

结论：

- few-shot 对结构化协议稳定性帮助非常大；
- 在 quick-test 上，这组结果已经构成当前最强的完整 black-box baseline。

### BML Local Qwen3-VL

#### direct + structured

- parse success rate: `8 / 65 = 12.31%`
- macro-F1: `0.071`

结论：

- 本地 Qwen 在直接结构化输出下同样不稳定；
- 问题首先表现为协议与解析失败，而不是单纯分类能力不足。

#### 5-shot + structured

- parse success rate: `65 / 65 = 100%`
- macro-F1: `0.489`

结论：

- few-shot 也能明显提升本地模型的结构化输出稳定性；
- 但当前本地 structured baseline 在 `rule4` 上仍然失效，说明 few-shot 解决了“格式问题”，没有完全解决“规则泛化问题”。

#### direct + classification_only

- parse success rate: `64 / 65 = 98.46%`
- macro-F1: `0.528`

结论：

- 当不要求 bbox 时，本地模型的分类能力可以较稳定地观测到；
- 当前最能反映“模型本体规则分类能力”的 baseline 是这组。

#### author-style 5-shot + classification_only

- parse success rate: `65 / 65 = 100%`
- macro-F1: `0.449`

结论：

- 这说明“模仿论文风格的 5-shot”并没有稳定优于 direct classification-only；
- 当前 few-shot 示例设计仍然会显著影响本地模型的类别偏置，特别是 `rule3` 和 `rule4`。

## 3. 当前结果说明了什么

### 结论一：few-shot 的主要价值首先是“稳定协议”

目前最明确的现象不是“few-shot 一定提高识别精度”，而是：

- `direct + structured` 容易解析失败；
- `5-shot + structured` 可以把 parse success rate 大幅拉高。

因此 few-shot 在当前阶段首先解决的是：

- JSON 输出是否合格；
- bbox 是否按要求返回；
- 多字段协议是否能被 parser 正确消费。

### 结论二：classification-only 更适合看模型本体能力

当前结构化输出同时要求：

- 规则判断；
- bbox；
- explanation；
- 多字段 JSON。

这会把“识别能力”和“输出协议能力”混在一起。

因此目前更适合比较模型分类能力的口径是：

- `classification_only`

这也是为什么本地 Qwen 的 `direct + classification_only` 比 `direct + structured` 更值得作为“模型能力观察点”。

### 结论三：当前本地模型已经表现出“高 recall、低 precision”趋势

在本地 Qwen 的 direct classification-only 结果中：

- rule1 的 recall 明显高于 precision；
- rule3 / rule4 也存在类似倾向。

这与作者论文中“VLM 倾向保守、容易过报”的总体观察是一致的。

只是当前 quick-test 子集是平衡构造的，因此 precision 不会像 full test 那样被长尾分布进一步压低。

### 结论四：当前的 author-style 5-shot 还不是“真正复现作者 5-shot”

需要特别说明：

- 当前代码中的 author-style 5-shot 只是**模仿论文风格**；
- 它不是对作者仓库 5-shot 推理格式与 few-shot 示例的严格复现。

主要差异包括：

- 当前输出使用 `violated_rule_ids + explanation + target_bbox` 聚合格式；
- 作者仓库原始 VQA 输出更接近稀疏字典格式；
- 当前 few-shot 示例固定为 `clean + rule1 + rule2 + rule3 + rule4`；
- 作者仓库示例中包含多违规样本。

因此，当前 author-style 结果更适合被解释为：

- “论文风格启发版 few-shot 对照”

而不是：

- “作者论文 5-shot baseline 的严格复现值”。

---

## 当前阶段的总体判断

### Point 1 已经完成的关键工作

- benchmark 读取与 schema 契约冻结；
- quick-test 子集冻结；
- API baseline 跑通；
- BML local Qwen baseline 跑通；
- direct / five-shot / structured / classification-only 四种组合已完成首轮比较；
- official-style export bridge 已补齐。

### Point 1 当前最值得保留的结论

1. 结构化输出比纯分类明显更难。
2. few-shot 对结构化协议稳定性帮助极大。
3. 本地 Qwen 的分类能力是存在的，但有明显“高 recall、低 precision”倾向。
4. 当前的论文风格 few-shot 还不能直接当作作者原始 baseline 复现。

### 当前最合理的下一步

1. 冻结 quick-test 结果，作为开题阶段的 baseline 证据。
2. 明确区分：
   - 我们自己的 structured baseline
   - 论文风格启发版 few-shot
3. 后续在 BML 平台上继续推进更贴近作者仓库的 baseline 复现。
4. 在 baseline 口径稳定后，进入 Point 1 主方法：
   - `candidate -> evidence -> predicate -> executor -> explanation`

---

## 开题汇报建议口径

如果需要在开题阶段做简要汇报，可以把当前工作总结为：

> 我们已经完成了 Point 1 的 benchmark 基础设施与第一阶段 baseline 搭建，初步验证了 ConstructionSite10k 四规则闭域任务在 black-box VLM 上是可运行的。当前结果表明，few-shot 对结构化输出稳定性帮助显著，本地 Qwen 也已展现出可观测的规则分类能力，但仍存在明显的高 recall、低 precision 倾向。下一步将继续在 BML 平台上强化 baseline 复现，并逐步转入 Point 1 的证据链主方法实现。
