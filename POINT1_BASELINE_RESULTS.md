# Point 1 Baseline Results Summary

更新时间：2026-04-14

本文档总结当前 Point 1 第一阶段 baseline 的实际实验结果与当前评测状态。
所有结果均基于闭域 quick-test 子集：

- `balanced_dev_15x5`：来自 train，用于固定 few-shot 示例
- `balanced_test_13x5`：来自 test，用于快速评估

说明：

- `balanced_test_13x5` 总计 65 张图
- 每个 bucket 13 张：
  - `clean`
  - `rule1`
  - `rule2`
  - `rule3`
  - `rule4`
- 之所以不是 `15x5`，是因为 official test split 中仅有 13 张纯净 `rule4` 单违规样本

---

## 1. 结果来源与运行环境

当前 Point 1 baseline 分成两条执行路径：

### API baseline

- 通过 OpenAI-compatible provider 调用远端模型
- 当前已跑通：
  - ModelScope `direct + structured`
  - ModelScope `5-shot + structured`

### BML 平台本地模型 baseline

- 通过 BML 平台加载本地 `Qwen3-VL-8B-Instruct`
- 当前文档中的 “Local Qwen” 指的是：
  - **模型在 BML 服务器本地加载**
  - **不是用户这台开发机本地运行**
- 当前文档记录的模型路径约定为：
  - `/home/bml/storage/qwen3_models`
- 当前已跑通：
  - BML Local Qwen `direct + structured`
  - BML Local Qwen `5-shot + structured`
  - BML Local Qwen `direct + classification_only`
  - BML Local Qwen `5-shot + classification_only`

其中：

- `structured`：要求完整结构化输出，包含 `target_bbox`
- `classification_only`：仅判断 rule violation，不要求 bbox，`target_bbox = null`

---

## 2. 当前已完成的 baseline 组合

### API baseline

1. ModelScope `direct + structured`
2. ModelScope `5-shot + structured`

### BML Local Qwen3-VL baseline

1. `direct + structured`
2. `5-shot + structured`
3. `direct + classification_only`
4. `5-shot + classification_only`

---

## 3. ModelScope structured baseline

模型：

- `Qwen/Qwen3-VL-8B-Instruct`

### 3.1 direct + structured

- parse success rate: `4 / 65 = 6.15%`
- macro-F1: `0.033`

Per-rule 指标：

| Rule | Precision | Recall | F1 |
|---|---:|---:|---:|
| rule1 | 0.500 | 0.077 | 0.133 |
| rule2 | 0.000 | 0.000 | 0.000 |
| rule3 | 0.000 | 0.000 | 0.000 |
| rule4 | 0.000 | 0.000 | 0.000 |

### 3.2 5-shot + structured

- parse success rate: `65 / 65 = 100%`
- macro-F1: `0.680`

Per-rule 指标：

| Rule | Precision | Recall | F1 |
|---|---:|---:|---:|
| rule1 | 0.667 | 0.308 | 0.421 |
| rule2 | 1.000 | 0.923 | 0.960 |
| rule3 | 0.556 | 0.769 | 0.645 |
| rule4 | 0.692 | 0.692 | 0.692 |

### 3.3 小结

对 ModelScope 而言：

- few-shot 对**结构化协议稳定性**帮助极大
- `5-shot + structured` 已经是一个可信、可复现实验对照组
- `direct + structured` 的主要问题不是完全不会分类，而是输出协议和 bbox 不稳定

---

## 4. BML Local Qwen3-VL structured baseline

模型：

- BML 本地 `Qwen3-VL-8B-Instruct`
- 路径约定：`/home/bml/storage/qwen3_models`

### 4.1 direct + structured

- parse success rate: `8 / 65 = 12.31%`
- macro-F1: `0.071`

Per-rule 指标：

| Rule | Precision | Recall | F1 |
|---|---:|---:|---:|
| rule1 | 0.375 | 0.231 | 0.286 |
| rule2 | 0.000 | 0.000 | 0.000 |
| rule3 | 0.000 | 0.000 | 0.000 |
| rule4 | 0.000 | 0.000 | 0.000 |

### 4.2 5-shot + structured

- parse success rate: `65 / 65 = 100%`
- macro-F1: `0.489`

Per-rule 指标：

| Rule | Precision | Recall | F1 |
|---|---:|---:|---:|
| rule1 | 0.364 | 0.615 | 0.457 |
| rule2 | 1.000 | 1.000 | 1.000 |
| rule3 | 0.545 | 0.462 | 0.500 |
| rule4 | 0.000 | 0.000 | 0.000 |

### 4.3 小结

BML 本地 Qwen3-VL 在 `direct + structured` 下明显不稳定，但 `5-shot + structured` 已经把结构化解析率拉到了 `100%`。

这说明：

1. few-shot 对本地结构化协议稳定性同样有效
2. 结构化生成不稳定并不是不可修复的问题
3. 当前 BML 本地 structured baseline 的主要短板已经从“解析失败”转成“rule4 识别失败”

换句话说，当前瓶颈不再只是：

- bbox 输出
- structured JSON 约束
- 多字段联合生成

还包括：

- rule4 few-shot 泛化
- rule1 的 precision 控制
- structured 情况下的类别间偏置

---

## 5. BML Local Qwen3-VL classification-only baseline

### 5.1 direct + classification_only

- parse success rate: `64 / 65 = 98.46%`
- macro-F1: `0.528`

Per-rule 指标：

| Rule | Precision | Recall | F1 |
|---|---:|---:|---:|
| rule1 | 0.176 | 0.692 | 0.281 |
| rule2 | 0.667 | 0.923 | 0.774 |
| rule3 | 0.375 | 0.692 | 0.486 |
| rule4 | 0.455 | 0.769 | 0.571 |

### 5.2 author-style 5-shot + classification_only

- parse success rate: `65 / 65 = 100%`
- macro-F1: `0.449`

Per-rule 指标：

| Rule | Precision | Recall | F1 |
|---|---:|---:|---:|
| rule1 | 0.467 | 0.538 | 0.500 |
| rule2 | 0.929 | 1.000 | 0.963 |
| rule3 | 0.600 | 0.231 | 0.333 |
| rule4 | 0.000 | 0.000 | 0.000 |

### 5.3 小结

classification-only 结果表明：

1. BML 本地 Qwen3-VL 的**规则分类能力是存在的**
2. `direct + classification_only` 已经呈现出明显的：
   - 高 recall
   - 低 precision
3. 对齐论文风格后的 author-style `5-shot` 并没有形成稳定正增益
4. `5-shot` 对：
   - rule1 有帮助
   - rule2 维持很强
   - rule3 明显退化
   - rule4 直接失效
5. 因此当前更可信的本地分类 baseline 仍是：
   - `direct + classification_only`

这说明：

> few-shot 示例设计对本地 Qwen3-VL 影响很大；当前 author-style 5-shot 更接近论文范式，但在本地模型上并未整体优于 direct classification-only baseline。

---

## 6. 当前总体结论

### 结论 1：structured 输出明显更难

不论是 API 模型还是 BML 本地模型，只要要求：

- bbox
- 解释
- 多字段结构化 JSON

baseline 稳定性都会明显下降。

### 结论 2：classification-only 更适合先测模型本体能力

当不要求 bbox 时：

- BML 本地 Qwen3-VL 的解析率接近 100%
- 分类能力可以被较稳定地观测到

### 结论 3：BML 本地 Qwen direct_cls 更接近“高 recall、低 precision”

尤其：

- rule1 的 recall 明显高于 precision
- rule3 / rule4 也有类似趋势

这和 ConstructionSite10k 论文中对 VLM baseline 的总体描述是相符的。

### 结论 4：当前最可信的 baseline 对照组

截至目前，最值得保留的 baseline 对照组是：

1. **ModelScope 5-shot + structured**
2. **BML Local Qwen 5-shot + structured**
3. **BML Local Qwen direct + classification_only**
4. **BML Local Qwen author-style 5-shot + classification_only**（作为 paper-aligned few-shot 对照）

前者代表：

- 当前较强的完整结构化 black-box baseline

中间这组代表：

- 当前 BML 平台上可复现的本地结构化 baseline

后两者代表：

- 当前较可信的本地模型分类 baseline
- 当前论文风格 few-shot 设计的实际效果

---

## 7. Full test（3004）结果：BML Local Qwen + author-style prompt

在 full test 上，我们额外补跑了：

1. **BML Local Qwen author-style direct + classification_only**
2. **BML Local Qwen author-style 5-shot + classification_only**

说明：

- 模型：`/home/bml/storage/qwen3_models`
- 评测集合：`test.parquet`（3004 张）
- prompt 口径：`author_vqa`
- few-shot 口径：**无泄漏版** `author_train_mimic`
- 输出类型：`classification_only`

### 7.1 Full test 解析情况

- direct parse success rate: `3004 / 3004 = 100%`
- 5-shot parse success rate: `3004 / 3004 = 100%`

这说明当前 author-style full test 路径已经稳定，不再像早期 structured baseline 那样主要受协议解析问题限制。

### 7.2 论文风格分 rule 表

> 表：BML Local Qwen3-VL 在 ConstructionSite10k full test 上的 author-style classification-only 结果

| Rule | Direct P | Direct R | Direct F1 | 5-shot P | 5-shot R | 5-shot F1 |
|---|---:|---:|---:|---:|---:|---:|
| rule1 | 0.197 | 0.481 | 0.280 | 0.206 | 0.917 | 0.336 |
| rule2 | 0.043 | 0.680 | 0.081 | 0.062 | 0.880 | 0.116 |
| rule3 | 0.052 | 0.651 | 0.096 | 0.086 | 0.286 | 0.132 |
| rule4 | 0.019 | 0.708 | 0.036 | 0.026 | 0.208 | 0.045 |
| macro-F1 | - | - | 0.123 | - | - | 0.157 |

### 7.3 Full test 小结

这组 full test 结果有几个很清楚的现象：

1. **5-shot 整体优于 direct**
   - macro-F1 从 `0.123` 提升到 `0.157`

2. **Rule 1 / Rule 2 的提升主要来自 recall 上升**
   - rule1 recall 从 `0.481` 提升到 `0.917`
   - rule2 recall 从 `0.680` 提升到 `0.880`
   - precision 仅小幅改善，说明模型仍有明显过报倾向

3. **Rule 3 / Rule 4 在 5-shot 下更保守**
   - rule3 recall 从 `0.651` 降到 `0.286`，但 precision 从 `0.052` 升到 `0.086`
   - rule4 recall 从 `0.708` 降到 `0.208`，但 precision 从 `0.019` 升到 `0.026`
   - 最终 F1 仍略有改善，但 rule4 依然是最薄弱规则

4. **full test 明确暴露了本地 Qwen 的“高 recall、低 precision”特征**
   - direct 尤其明显，rule2 / rule3 / rule4 大量误报
   - 5-shot 虽然改善了整体 F1，但没有从根本上解决 precision 偏低的问题

因此，这组结果非常适合作为 Point 1 主方法的动机：

> black-box VLM baseline 虽然能报出较多违规，但 precision 控制较差，尤其在复杂规则上容易过报，因此需要显式的 `candidate -> predicate -> executor -> explanation` 证据链方法来约束决策。

---

## 8. 当前评测出口状态

当前仓库已经具备两层结果消费能力：

### 8.1 内部 summary / comparison

用于快速看：

- parse success rate
- bucket hit rate
- per-rule precision / recall / F1
- macro-F1

对应脚本：

- `scripts/analyze_point1_baselines.py`

### 8.2 official eval bridge

仓库现在已经补上了一个最小 official eval bridge。
它的作用不是“重新评分”，而是：

> 把仓库内部的 Point 1 结构化预测，导出成 ConstructionSite10k 官方风格预测 JSON。

当前状态：

- 已完成格式导出
- 已补测试
- 已可作为后续接官方评测仓库的稳定出口
- **尚未**在本仓库内完成 “调用官方仓库脚本并产出最终官方分数” 的闭环

因此目前应理解为：

- `official format export` 已完成
- `official full evaluation pipeline` 还未完全打通

---

## 9. 下一步建议

1. 将这组 full test 结果整理为论文主表候选版本，并补充正式表题/图注
2. 专门分析 Rule 4 在 direct / 5-shot 下 precision 极低的原因
3. 用新的 official eval bridge，把现有 baseline 统一导出成官方风格预测文件
4. 明确记录哪些结果用于论文主表，哪些结果只作为 error analysis
5. 在 baseline 口径稳定后，再进入 Point 1 主方法第一条真实链路：
   - Rule 1 `candidate -> predicate -> executor -> explanation`

---

## 10. 当前建议保留的核心结果文件

建议保留：

- `artifacts/point1/direct-modelscope-balanced_test_13x5-repair.json`
- `artifacts/point1/direct-modelscope-balanced_test_13x5-repair.summary.json`
- `artifacts/point1/fiveshot-modelscope-balanced_test_13x5.json`
- `artifacts/point1/fiveshot-modelscope-balanced_test_13x5.summary.json`
- `artifacts/point1/modelscope-balanced_test_13x5-comparison-repair.json`
- `artifacts/point1/direct-localqwen-balanced_test_13x5.json`
- `artifacts/point1/direct-localqwen-balanced_test_13x5.summary.json`
- `artifacts/point1/direct-localqwen-balanced_test_13x5.official.json`
- `artifacts/point1/direct-localqwen-balanced_test_13x5.eval-summary.json`
- `artifacts/point1/fiveshot-localqwen-balanced_test_13x5.json`
- `artifacts/point1/fiveshot-localqwen-balanced_test_13x5.summary.json`
- `artifacts/point1/fiveshot-localqwen-balanced_test_13x5.official.json`
- `artifacts/point1/fiveshot-localqwen-balanced_test_13x5.eval-summary.json`
- `artifacts/point1/directcls-localqwen-balanced_test_13x5.json`
- `artifacts/point1/directcls-localqwen-balanced_test_13x5.summary.json`
- `artifacts/point1/directcls-localqwen-authorvqa-fulltest.json`
- `artifacts/point1/directcls-localqwen-authorvqa-fulltest.summary.json`
- `artifacts/point1/fiveshotcls-localqwen-balanced_test_13x5.json`
- `artifacts/point1/fiveshotcls-localqwen-balanced_test_13x5.summary.json`
- `artifacts/point1/fiveshotcls-localqwen-authorvqa-fulltest.json`
- `artifacts/point1/fiveshotcls-localqwen-authorvqa-fulltest.summary.json`
- `artifacts/point1/localqwen-cls-comparison.json`
- `artifacts/point1/localqwen-authorvqa-fulltest-comparison.json`
- `artifacts/point1/localqwen-structured-comparison.json`

如果后续开始统一导出官方风格预测文件，建议并列保留：

- `*.official.json`
- `*.eval-summary.json`
