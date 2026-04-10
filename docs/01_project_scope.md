# 01 Project Scope

## 论文总体结构
本课题采用“两个理论研究点 + 一个系统实现点”的固定结构：
- Point 1：闭域 benchmark 上的证据链驱动可解释违规识别。
- Point 2：知识增强、RAG 与 PEFT 协同的跨场景鲁棒性方法。
- Point 3：原型系统与工程验证。

## 为什么必须收窄题目
本项目不再写成泛化的“基于多模态大模型的施工安全隐患识别研究”，而是收敛为：
**面向施工场景典型规则违规的知识增强可解释视觉语言隐患识别研究——以 PPE、高处作业与人机交互为例。**

这样做的目的：
- 固定论文主体三章，避免开题后继续发散。
- 让 Point 1 与 Point 2 的研究问题边界清楚。
- 让 Point 3 成为验证前两点的系统闭环，而不是单独的杂项工程。

## Point 1 的正式边界
- 数据：ConstructionSite10k
- 规则：仅四条内置规则
- 输入：单张图像
- 输出：结构化 JSON，包含 decision / bbox / evidence / explanation
- 不做：RAG、LoRA/QLoRA、外部国标检索、视频主实验

## Point 2 的正式边界
- 研究对象：规则知识增强、相似案例检索、RAG、LoRA/QLoRA / PEFT
- 核心目标：提升小样本与跨场景条件下的 precision、grounding IoU 与 explanation consistency
- 对照组：zero-shot、Prompt/CoT、RAG-only、LoRA-only、LoRA+RAG

## Point 3 的角色
- 不是单独追求产品功能，而是把 Point 1 / Point 2 的方法串成巡检闭环：
  图像/视频输入 -> 候选筛查 -> 规则检索 -> 违规定位 -> 报告生成 -> 人工复核 -> 难例回流
