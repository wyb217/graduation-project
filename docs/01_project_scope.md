# 01 Project Scope

## 论文总体结构

本课题采用“两个研究点 + 一个系统实现点”的固定结构：

- Point 1：ConstructionSite10k 四规则闭域、单图、证据链驱动、可解释违规识别。
- Point 2：面向知识源变化与规则表述变化的知识增强 RAG / PEFT 协同方法。
- Point 3：原型系统与工程闭环验证。

## 当前阶段判断

目前最准确的项目状态不是“整个系统已完成”，而是：

- Point 1 已经推进到 Rule 1 主方法的 full test 阶段性结果；
- Point 2 已经完成问题收敛与边界重写，但尚未进入稳定实现与正式结果阶段；
- Point 3 仍然只是后续的接口消费层，不应提前反向侵入研究代码。

## 为什么必须收窄题目

项目不再写成泛化的“基于多模态大模型的施工安全隐患识别研究”，而是收敛为：

**面向施工场景典型规则违规的知识增强可解释视觉语言隐患识别研究。**

收窄目的有三点：

- 固定论文主体结构，避免开题后继续发散；
- 让 Point 1 与 Point 2 的研究问题和证据链清楚分离；
- 让 Point 3 成为前两点稳定接口的工程验证，而不是与研究主线混写。

## Point 1 的正式边界

- 数据：ConstructionSite10k
- 规则：仅四条内置规则
- 输入：单张图像
- 输出：结构化 JSON，包含 decision / bbox / evidence / explanation
- 允许：局部 crop、candidate generation、predicate extraction、rule executor、显式 explanation slots
- 不做：RAG、LoRA/QLoRA、外部规范检索、视频主实验

当前现实进度补充：

- Rule 1 已有 full test 阶段性结果；
- Rule 2 / Rule 3 / Rule 4 仍然属于后续推进内容；
- 因此 Point 1 当前可以声称的是“Rule 1 主线已成立”，不能扩大成“Point 1 全部四条规则都已完成”。

## Point 2 的正式边界

- 研究对象：规则知识单元、企业制度、国标条款、相似案例、RAG、LoRA/QLoRA / PEFT
- 核心目标：在知识源变化与规则表述变化下，提升违规识别的 precision、grounding IoU、explanation consistency 与可追溯性
- 辅助分析：必要时可以把 illumination 等数据集属性作为分层分析维度，但它们不是 Point 2 的主 claim
- 对照组：prompt-only、RAG-only、PEFT-only、RAG+PEFT

明确限制：

- 在没有额外数据和明确操作化定义时，不把 Point 2 笼统写成“跨场景鲁棒性”；
- Point 2 可以借用 Point 1 的 benchmark / schema / eval 契约，但不能破坏 Point 1 的闭域 benchmark 口径。

## Point 3 的角色

Point 3 不是单独追求功能堆砌，而是把 Point 1 / Point 2 的稳定接口串成工程闭环，例如：

```text
图像输入 -> 规则候选识别 -> 结果展示/复核 -> 结果归档 -> 难例回流
```

Point 3 只消费稳定接口，不反向侵入 Point 1 / Point 2 的研究实现。
