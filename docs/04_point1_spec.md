# 04 Point 1 Spec

## 一句话定义
Point 1 研究：
**面向 ConstructionSite10k 四规则闭域的证据链驱动可解释违规识别。**

## 目标
把违规识别从“整图问答”改写成：
`candidate -> evidence -> rule executor -> explanation`

## 输入
- 单张 ConstructionSite10k 图像
- 可选：规则 ID / 全规则扫描模式

## 输出
每个违规实例输出一个结构化对象：
```json
{
  "rule_id": 1,
  "decision_state": "violation",
  "target_bbox": [0.22, 0.59, 0.28, 0.75],
  "supporting_evidence_ids": ["e12", "e15"],
  "counter_evidence_ids": ["e08"],
  "unknown_items": ["head_visibility"],
  "reason_slots": {
    "subject": "worker on the left",
    "missing_item": "hard hat",
    "scene_condition": "on foot at the site"
  },
  "reason_text": "The worker on the left is on foot at the site and is not wearing a hard hat.",
  "confidence": 0.84
}
```

## 四规则分解
### Rule 1
- Candidate：person + body part crops
- Predicates：hard_hat / upper_body_covered / lower_body_covered / toe_covered / visibility
- 主输出：person bbox
- 当前实现注记：
  - 现阶段仓库已具备 Rule 1 的第一条主方法链路；
  - 当前默认实现是 `candidate + heuristic predicates + symbolic executor`；
  - 当前输出仍是 per-candidate prediction，尚未完成 image-level aggregation。

### Rule 2
- Candidate：person + local context
- Predicates：elevated_context / edge_nearby / edge_protected / harness_visible
- 主输出：person bbox
- 注意：`>=3m` 只能作为 proxy reasoning，不声称物理真值

### Rule 3
- Candidate：edge / pit / opening region
- Predicates：accessible_edge_region / edge_protected / warning_present
- 主输出：risk-region bbox

### Rule 4
- Candidate：person-excavator pair
- Predicates：excavator_present / in_risk_zone_proxy / separation_barrier_present / relative_distance / relative_side
- 主输出：person bbox（允许附带 excavator support）

## Point 1 明确禁止项
- 外部规范检索
- RAG
- LoRA / QLoRA / PEFT
- 额外行业标准扩展
- 视频主实验
- 自由文本报告作为主输出

## 主评测指标
优先级 P1：
- Per-rule Precision
- Macro-F1
- IoU
- Acc@0.5
- TAR@0.5

优先级 P2：
- Explanation faithfulness
- Citation validity
- Stratified metrics by illumination / camera_distance / view / quality_of_info

## 实现顺序
1. benchmark 与 schema
2. stable baseline
3. Rule 1 真实视觉路径
4. Rule 4 pair reasoning
5. Edge module
6. Rule 2 / Rule 3
7. executor + explanation hardening
8. official bridge + ablations + error analysis

## 当前策略说明：VLM 在 Point 1 中的位置

当前 Point 1 里，VLM 已经承担 baseline 对照角色：

- direct / zero-shot
- 5-shot
- author-style prompt

但在主方法链路中，VLM 目前还没有直接进入 Rule 1 predicate extraction。

这样做的原因是：

- Point 1 的核心研究目标不是“再做一个统一黑箱判别器”；
- 而是把识别过程改写成显式的：

`candidate -> evidence/predicate -> rule executor -> explanation`

因此后续更推荐的演进方向不是：

- `candidate -> unified VLM final decision`

而是：

- `candidate -> unified VLM predicate extraction -> executor -> explanation`

换句话说，VLM 更适合作为 **统一的局部谓词判别器**，而不是最终 rule decision 的唯一来源。
