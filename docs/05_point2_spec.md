# 05 Point 2 Spec

## 一句话定义

Point 2 研究：
**面向施工典型规则违规识别中的知识源变化与规则表述变化，研究知识增强 RAG 与 PEFT 的协同稳健性方法。**

## 为什么要改写 Point 2

Point 2 之前如果笼统写成“跨场景鲁棒性”，会有一个明显问题：

- 如果没有额外场景数据，也不愿意把 ConstructionSite10k 属性字段当主轴，那么“跨场景”就很难被操作化定义。

因此当前更合理的 Point 2 表述是：

- 先把问题收紧成 **知识变化** 与 **规则表述变化** 下的稳健识别；
- 必要时把 illumination 等条件作为辅助分层分析；
- 先把主 claim 写成可评测问题，再考虑是否扩展到更大的场景迁移。

## 核心研究问题

1. 如何把国标条款、企业制度、相似案例从“原始文本”整理成可检索、可落地、可追溯的知识单元？
2. 当知识来源或规则表述发生变化时，RAG-only、PEFT-only、RAG+PEFT 各自能带来什么收益？
3. 如何在引入外部知识后，同时提升 precision、grounding IoU、解释一致性和引用可追溯性，而不是仅仅“换更大模型”？

## Point 2 的主评测维度

### 1. 知识源变化稳健性

同一类违规规则，分别来自：

- 国标条文
- 企业制度
- 相似案例描述
- 结构化规则模板

目标是看模型在知识来源变化时，判断是否仍稳定、有依据、少误报。

### 2. 规则表述变化稳健性

同一规则语义可能以不同表达形式出现，例如：

- 条款原文
- 改写后的工程语言
- 企业口径措辞
- 案例叙述语言

Point 2 要评估模型在这些表述变化下，是否还能稳定对齐到同一违规语义。

### 3. 辅助视觉条件分析

- illumination
- camera_distance
- view
- quality_of_info

这些属性可以作为辅助分层维度使用；
但当前不应把它们写成 Point 2 的主 claim，更不应在没有额外证据时把它们泛化成“跨场景鲁棒性”的全部定义。

## 与 Point 1 的边界

Point 2 可以复用：

- benchmark loader
- 通用 schema
- 通用评测模块
- 通用 artifact 导出逻辑

Point 2 不得直接复用：

- Point 1 的闭域 rule executor 作为核心方法
- Point 1 的内部候选生成逻辑作为黑箱捷径

如果确需使用 Point 1 输出，只能通过：

- 已持久化的 JSON artifact
- 公共接口协议
- 明确的 adapter 层

## Point 2 的知识单元设计

Point 2 不应把整段原始规范直接塞进 prompt，而应先把知识整理成可检索单元，例如：

- `clause_id`
- `source_type`
- `source_name`
- `rule_family`
- `normalized_rule_text`
- `visual_conditions`
- `negative_constraints`
- `suggested_actions`

这些知识单元应满足：

- 可版本化
- 可序列化
- 可检索
- 可和视觉证据或输出解释对齐

## 建议主线

### 知识层

- 规则条款模板化
- 企业制度结构化
- 相似案例索引
- 知识版本与来源管理

### 模型层

- LoRA / QLoRA / PEFT
- 面向规则识别的数据构造与训练集组织

### 推理层

- RAG 输入组装
- 模板化输出
- 规则一致性约束
- retrieval trace / matched clause 暴露

### 评测层

- prompt-only
- RAG-only
- PEFT-only
- RAG+PEFT

## Point 2 主输出

- rule_id / category
- matched_clause_id
- matched_rule_text
- evidence_box
- reason
- suggestion
- retrieval_trace（启用 RAG 时）
- confidence

## Point 2 主评测

- Precision / Recall / F1
- Grounding IoU
- Explanation consistency
- Citation validity / retrieval usefulness
- Robustness gap across knowledge-source variants
- Robustness gap across rule-expression variants

## Point 2 明确不该声称的内容

- 在没有额外数据和明确定义时，不把 Point 2 直接声称为广义“跨场景鲁棒性”研究；
- 不把 illumination 等辅助属性分析冒充为完整的场景迁移结论；
- 不把 Point 2 写成“RAG 和微调让模型更强”这种不可评测口号。

## Point 2 代码组织建议

- `src/knowledge/`：规则知识库、案例库、检索器
- `src/finetune/`：数据构造、LoRA、训练器
- `src/point2/`：推理与实验编排

## 开发顺序建议

1. 定义知识单元 schema 与版本化方式
2. 做 prompt-only / RAG-only baseline
3. 做 PEFT-only baseline
4. 最后做 RAG+PEFT
5. 再整理 robustness matrix 与结果叙事
