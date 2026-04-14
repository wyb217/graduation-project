# 09 Point 1 Foundation Status

## 本轮完成内容

已完成 Point 1 的第一阶段基础设施实现：

1. Python 项目骨架与 `src/` 布局；
2. `common.schemas.NormalizedBBox`；
3. `common.schemas.Point1Prediction` / `Point1Evidence`；
4. ConstructionSite10k typed sample / violation / attributes schema；
5. annotation parser；
6. JSON / parquet dataset loader；
7. split registry 读取接口；
8. bbox / schema / parser / loader / registry 测试；
9. README 的环境与命令说明；
10. 冻结的 `balanced_15x5` 快速测试子集。
11. conda-first 环境说明与初学者友好的开发脚本。
12. Point 1 API baseline（direct / 5-shot）骨架与 provider 配置入口。
13. Rule 1 轻量真实视觉主链路（person candidate -> PPE predicates -> executor -> explanation）。

## 当前代码边界

- Point 1 已接入：
  - black-box API baseline
  - BML 本地 Qwen baseline
  - Rule 1 的本地 / 远端 VLM predicate 后端
- Point 1 仍明确禁止：RAG、LoRA、QLoRA；
- Point 2 仅保留目录边界，不含实现；
- Point 3 仅保留系统层占位目录。

## Rule 1 当前状态

当前仓库已经增加 Rule 1 的第一条方法链路：

- person candidate generation
- Rule 1 predicate extraction
- Rule 1 executor
- Rule 1 explanation mapping
- image-level aggregation
- 26 张 `clean + rule1` 小闭环 runner / summary

当前实现定位：

- 不是最终论文精度版本；
- 但已经具备真实视觉输入、明确模块边界与单元测试；
- 可作为后续 Rule 1 专项实验和 executor 扩展的起点。

## Rule 1 pipeline 当前实现

当前 Rule 1 pipeline 的执行顺序为：

1. `candidate_generator` 生成 person candidates；
2. `predicate_extractor` 对每个候选输出 Rule 1 predicate bundle；
3. `executor` 根据 predicates 做显式规则判断；
4. `explanation` 模块把决策结果转成 `reason_slots` 与 `reason_text`。

当前默认实现分别是：

- candidate generation：OpenCV HOG people detector
- predicate extraction：基于局部 crop 的 heuristic 判别
- executor：显式 `violation / no_violation / unknown` 规则
- explanation：模板化结构化解释

当前仓库还增加了一个实验性的替代后端：

- candidate generation：OpenCV HOG
- predicate extraction：OpenAI-compatible VLM over person crop
- executor / explanation：保持不变

现在还额外支持：

- candidate generation：OpenCV HOG
- predicate extraction：BML 本地 Qwen3-VL over person crop
- executor / explanation：保持不变

当前 BML 侧的推荐默认接入方式是：

- **优先使用本地 Qwen3-VL**
- 远端 ModelScope / OpenAI-compatible provider 主要保留为对照组或补充实验

也就是说，Rule 1 当前已经可以比较：

1. `candidate -> heuristic predicates -> executor -> explanation`
2. `candidate -> VLM predicates -> executor -> explanation`
3. `candidate -> local Qwen predicates -> executor -> explanation`

当前 VLM predicate 版内部已经加入两个精度 gate：

- `ppe_applicable`：该候选是否属于 Rule 1 真正关心的 on-foot worker
- `head_region_visible`：头部区域是否足够可见，能否可靠判断 hard hat

其中 `ppe_applicable` 必须理解为 **candidate-local**：

- 它只判断当前候选是否是 Rule 1 应检查的 worker；
- 它不是 image-level 排他开关；
- 当前候选即使同时也违反 Rule 2 / 3 / 4，Rule 1 仍然可以成立。

这些 gate 的目的不是提高 recall，而是先压：

- machine operator / cab occupant 被误算成 PPE 违规
- 头部不可见却被误判为未戴安全帽

这里的关键点是：

- 当前内部仍保留 **per-candidate prediction** 作为中间结果；
- 但对外已经补上 image-level aggregation，可输出单图级 Rule 1 最终判断；
- `unknown` 是一等状态，不会在可见性不足时被强行压成 yes/no。

## Rule 1 explain 当前实现

当前 explain 不是额外让模型自由生成，而是由 executor 结果直接映射得到：

- `reason_slots` 至少包含 `subject`、`missing_item`、`scene_condition`
- `reason_text` 由模板函数根据 `decision_state`、`missing_items`、`unknown_items` 生成

因此当前 explanation 的性质是：

- 可审计
- 与 predicates / executor 保持一致
- 更适合 Point 1 的 evidence-chain 口径

但它也意味着：

- 当前 explanation 偏模板化；
- 语言自然度不是第一目标；
- 后续如需增强，可在不替换 executor 的前提下增加受约束的 verbalization。

## Point 1 中 VLM 的当前角色与后续位置

当前仓库里，VLM 已经用于 Point 1 baseline：

- direct
- 5-shot
- author-style prompt

但当前 Rule 1 主方法链路 **还没有直接接入 VLM**。目前主方法更像：

- lightweight detector + heuristic predicates + symbolic executor

这样做的目的，是先把：

- candidate
- predicate
- executor
- explanation

这四层的结构和协议稳定下来，而不是一开始就退回黑箱最终判别。

当前更推荐的下一步不是“统一 VLM 最终判别”，而是：

- 把 heuristic predicate extractor 升级成 **统一的 VLM predicate judge**
- 保留 executor / explanation 作为显式规则层

也就是说，后续更理想的 Point 1 路线是：

`candidate -> unified VLM predicate extraction -> executor -> explanation`

当前这条路线的近期重点已经进一步收窄为：

- 先在 26 张 `clean + rule1` 上验证 gate 是否不把结果做坏
- 再在 65 张 `balanced_test_13x5` 上看 cross-rule false positives 是否下降

## parquet 数据支持

当前 benchmark 层已经支持直接读取 ConstructionSite10k 的 parquet 文件。

设计选择：

- 复用同一套 typed schema，不为 parquet 单独造第二套协议；
- parquet 中的 `image` 字段解析为 `SampleImage`；
- 默认只保留 `image.path`，不默认保留 `image.bytes`，避免把整套数据集图片二进制一次性塞进内存；
- 如后续 Point 1 baseline 需要直接从 parquet 取图像字节，可显式开启 `include_image_bytes=True`。

## 关键文件

- `src/common/schemas/bbox.py`
- `src/common/schemas/point1.py`
- `src/benchmark/constructionsite10k/types.py`
- `src/benchmark/constructionsite10k/parser.py`
- `src/benchmark/constructionsite10k/loader.py`
- `src/benchmark/constructionsite10k/registry.py`
- `src/benchmark/constructionsite10k/subsets.py`
- `src/benchmark/splits/constructionsite10k_balanced_15x5.json`
- `src/benchmark/splits/constructionsite10k_balanced_dev_15x5.json`
- `src/benchmark/splits/constructionsite10k_balanced_test_13x5.json`
- `src/point1/baselines/`
- `src/point1/candidates/`
- `src/point1/predicates/`
- `src/point1/executor/`
- `src/point1/explanation/`
- `src/point1/pipelines/rule1.py`
- `scripts/run_point1_api_baseline.py`
- `configs/system/providers.example.json`

## 快速测试子集说明

已增加一个冻结子集：

- 名称：`balanced_15x5`
- 来源：train split
- 选择策略：
  - `clean`：四条规则都为 `null`
  - `rule1~rule4`：仅该单条规则违规
  - 多违规样本不进入该子集
  - 每桶按 `image_id` 排序后取前 15

因此总数是 **75**，不是 60。

## 下一阶段建议

1. 先在 `balanced_test_13x5_clean + balanced_test_13x5_rule1` 上跑通 Rule 1 小闭环并记录结果；
2. 再把 Rule 1 扩到 `balanced_test_13x5` 全 65 张，观察其它规则样本上的负例压力；
3. 进入 Rule 4 pair reasoning 与 edge-related modules；
4. 继续补强 stratified metrics、error analysis 与 failure export。
