# 05 Point 2 Spec

## 一句话定义
Point 2 研究：
**面向施工典型规则违规识别的知识增强、RAG 与 PEFT 协同方法。**

## 研究问题
- 如何把施工规范、企业制度、相似案例显式接入模型，而不是只依赖参数记忆？
- 如何在小样本与跨场景条件下提升 precision、grounding IoU 与 explanation consistency？
- RAG-only、LoRA-only、LoRA+RAG 三者的收益边界分别是什么？

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

## 建议主线
- 知识层：规则条款模板化、相似案例索引、RAG 检索
- 模型层：LoRA / QLoRA / PEFT
- 推理层：模板化输出、证据链、规则一致性约束
- 评测层：zero-shot / Prompt / CoT / RAG-only / LoRA-only / LoRA+RAG

## Point 2 主输出
- rule_id / category
- clause / matched rule text
- evidence_box
- reason
- suggestion
- retrieval trace（如果启用 RAG）
- confidence

## Point 2 主评测
- Precision / Recall / F1
- Grounding IoU
- BERTScore / explanation consistency
- Stratified robustness
- Retrieval usefulness（如果启用 RAG）

## Point 2 代码组织建议
- `src/knowledge/`：规则知识库、案例库、检索器
- `src/finetune/`：数据构造、LoRA、训练器
- `src/point2/`：推理与实验编排

## 开发注意事项
- 先实现 RAG-only 与 LoRA-only，最后再做 LoRA+RAG。
- 所有外部知识源必须有版本号、来源说明与序列化格式。
- 训练数据构造脚本必须可复跑。
