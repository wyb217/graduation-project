# 04 Point 1 Spec

## 一句话定义

Point 1 研究：
**面向 ConstructionSite10k 四规则闭域的证据链驱动可解释违规识别。**

## 当前已经验证到什么程度

Point 1 当前最可信的阶段性事实是：

- Rule 1 主方法已经从 quick-test 推进到 full test（3004）阶段性结果；
- 当前最稳定的 Rule 1 主线是：

```text
hog_then_torchvision -> local_qwen predicate -> executor -> explanation
```

- 这意味着 Point 1 已不再只是黑箱 baseline；
- 但这不等于四条规则都已完成，同等级结果目前只发生在 Rule 1。

## 总体目标

Point 1 要把违规识别从“整图黑箱问答”改写成：

```text
candidate -> predicate / evidence -> executor -> explanation
```

核心要求：

- 保持闭域 benchmark 口径；
- 输出结构化 JSON；
- 保持链路可审计，而不是重新退回自由文本最终判别。

## 输入

- 单张 ConstructionSite10k 图像
- 可选：指定 `rule_id` 或全规则扫描模式

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

## 当前 Rule 1 主线

### 默认口径

- candidate backend：`hog_then_torchvision`
- predicate backend：`local_qwen`
- executor：显式规则执行
- explanation：模板化 explanation slots + reason text

### 当前阶段性结果

截至目前，Rule 1 默认 full test 主线结果为：

- precision：`0.633`
- recall：`0.549`
- F1：`0.588`
- unknown rate：`0.572`

`crop_padding_profile=rule1_ppe` 变体当前更适合定位为：

- precision-oriented / unknown-reduction variant

而不是默认主线替代品。

## 四规则任务分解

### Rule 1

- Candidate：person + local crop
- Predicates：hard_hat / upper_body_covered / lower_body_covered / toe_covered / visibility / ppe_applicable
- 主输出：person bbox
- 当前状态：已有 full test 阶段性结果

### Rule 2

- Candidate：person + local context
- Predicates：elevated_context / edge_nearby / edge_protected / harness_visible
- 主输出：person bbox
- 说明：`>=3m` 只允许作为 proxy reasoning，不声称物理真值
- 当前状态：尚未形成同等级主结果

### Rule 3

- Candidate：edge / pit / opening region
- Predicates：accessible_edge_region / edge_protected / warning_present
- 主输出：risk-region bbox
- 当前状态：尚未形成同等级主结果

### Rule 4

- Candidate：person-excavator pair
- Predicates：excavator_present / in_risk_zone_proxy / separation_barrier_present / relative_distance / relative_side
- 主输出：person bbox，可附带 excavator support
- 当前状态：尚未形成同等级主结果

## 评测层级

### Quick-test

- `balanced_dev_15x5`：来自 train，用于固定 few-shot 示例
- `balanced_test_13x5`：来自 test，用于快速比较与 smoke test

### Full test

- `test.parquet`：用于正式阶段性判断

规则：

- subset 结果不能替代 full test 主结论；
- quick-test 主要用于方向判断和 prompt / parser / pipeline 快速对照。

## 主评测指标

优先级 P1：

- Per-rule Precision / Recall / F1
- Macro-F1
- IoU
- Acc@0.5
- TAR@0.5

优先级 P2：

- Explanation faithfulness
- Citation validity
- Stratified metrics by illumination / camera_distance / view / quality_of_info
- unknown rate / parse success rate

## Point 1 明确禁止项

- 外部规范检索
- RAG
- LoRA / QLoRA / PEFT
- 额外规则体系扩展
- 视频主实验
- 自由文本报告作为主输出

## 当前可以声称什么 / 不可以声称什么

可以声称：

- Rule 1 主方法已经有 full test 阶段性结果；
- 当前主线相对 black-box classification baseline 在 precision 与 controllability 上更有价值；
- detector fallback 已经证明有效。

不可以声称：

- Point 1 四条规则已经全部完成；
- Rule 2 / 3 / 4 已达到与 Rule 1 相同成熟度；
- 仅凭 quick-test 结果就替代正式主表结论。

## 当前优先级

1. Rule 1 full test error anatomy
2. unknown 压缩与 selective padding 判断
3. black-box baseline 的 parser / summary 修复与离线重评分
4. Rule 4 pair reasoning
5. Rule 2 / Rule 3 的 edge-related 模块
