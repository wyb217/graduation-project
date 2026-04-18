# Point 1 Baseline Results Summary

更新时间：2026-04-16

本文档总结当前 Point 1 第一阶段 baseline 与 Rule 1 主方法的实际实验结果。
本文档同时覆盖：

- quick-test 子集结果
- Rule 1 full test（3004）结果
- crop padding 变体对照

需要特别说明：

- 根目录保留的是结果总表；
- 更细的运行产物与 JSON artifact 当前位于 `.worktrees/feature-point1-foundation/artifacts/point1/`；
- 本文件里的“当前阶段判断”优先于旧版阶段叙述。

quick-test 相关结果主要基于闭域子集：

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

## 0. 当前阶段判断

截至当前，可以做如下阶段性判断：

1. **Point 1 中 Rule 1 主方法已经有正式 full test 阶段性结果**
   - 默认主线：`hog_then_torchvision + local_qwen + executor + explanation`
   - 默认 full test 配置：`crop_padding_profile=none`

2. **当前最可信的结论不是“Point 1 已整体完成”，而是“Rule 1 主线已经成立”**
   - 它已经不只是 quick-test 原型
   - 但还不能外推到 Rule 2 / 3 / 4 已成熟

3. **`crop_padding_profile=rule1_ppe` 值得保留，但当前不应替换默认主线**
   - 它能降低 FP，也能压缩一部分 unknown
   - 但在 full test 上没有带来更高的总体 F1

4. **下一步重点应是整理与收束，而不是继续无边界扩实验**
   - 先做 Rule 1 full test error anatomy
   - 再做 unknown 压缩与 selective padding 判断
   - 之后再考虑是否切向 Point 2

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

## 7. 当前评测出口状态

当前仓库已经具备两层结果消费能力：

### 7.1 内部 summary / comparison

用于快速看：

- parse success rate
- bucket hit rate
- per-rule precision / recall / F1
- macro-F1

对应脚本：

- `scripts/analyze_point1_baselines.py`

### 7.2 official eval bridge

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

## 8. 下一步建议

1. 先做 Rule 1 full test 的 FP / FN / unknown error anatomy
2. 修 black-box baseline 的 parser / summary，并基于已有 raw outputs 做离线重评分
3. 明确区分：
   - 哪些结果可进入正式主表
   - 哪些结果只适合作为探索性对照或 error analysis
4. 在 Rule 1 主线收束后，再推进：
   - Rule 4 pair reasoning
   - Rule 2 / Rule 3 的 edge-related 模块
   - Point 2 baseline

---

## 9. 当前建议保留的核心结果文件

根目录默认保留：

- `POINT1_BASELINE_RESULTS.md`
- `CONSTRUCTIONSITE10K_DATASET_STATS.md`

worktree 中默认保留：

- `.worktrees/feature-point1-foundation/artifacts/point1/*.summary.json`
- `.worktrees/feature-point1-foundation/artifacts/point1/*.official.json`
- `.worktrees/feature-point1-foundation/artifacts/point1/*.eval-summary.json`
- `.worktrees/feature-point1-foundation/artifacts/point1/*.progress.json`
- `.worktrees/feature-point1-foundation/artifacts/point1/*.checkpoint.json`
- `.worktrees/feature-point1-foundation/artifacts/point1/*.failures.json`

其中：

- `*.summary.json` / `*.official.json` / `*.eval-summary.json` 更接近正式结果汇总；
- `*.progress.json` / `*.checkpoint.json` / `*.failures.json` 更偏运行可观测性与排错材料；
- 这些文件可以分级使用，但在没有完成价值确认之前不应随意删除。
