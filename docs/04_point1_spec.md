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
