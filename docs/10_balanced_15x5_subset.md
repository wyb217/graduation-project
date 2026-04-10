# 10 Balanced 15x5 Subset

## 为什么不是 balance60

如果采用：

- Rule 1
- Rule 2
- Rule 3
- Rule 4
- clean

并且每一类各取 15 张，那么总数是：

`5 × 15 = 75`

所以这里把快速测试子集明确命名为 `balanced_15x5`，避免名字和实际样本数不一致。

## 用途

这个子集主要用于：

- 快速跑通 Point 1 baseline
- 调试结构化 JSON 输出
- 检查 rule executor / explanation 链路
- 做轻量人工抽查

它**不是**正式论文主评测集，也不替代完整 train/test 口径。

## 选择规则

来源数据：ConstructionSite10k train parquet shards。

桶定义：

1. `clean`
   - `rule_1_violation` ~ `rule_4_violation` 全部为 `null`
2. `rule1`
   - 只有 `rule_1_violation` 非空
3. `rule2`
   - 只有 `rule_2_violation` 非空
4. `rule3`
   - 只有 `rule_3_violation` 非空
5. `rule4`
   - 只有 `rule_4_violation` 非空

额外约束：

- 多违规样本不纳入该子集；
- 每个桶按 `image_id` 升序排序；
- 每个桶取前 15 张；
- 最终 registry 冻结到文件，不通过命令行临时采样。

## 文件位置

- 冻结 registry：`src/benchmark/splits/constructionsite10k_balanced_15x5.json`
- 生成脚本：`scripts/build_balanced_subset_registry.py`
- 生成逻辑：`src/benchmark/constructionsite10k/subsets.py`

说明：后续 baseline 阶段又增加了更贴合 API 调试流程的：

- `balanced_dev_15x5`
- `balanced_test_13x5`

它们分别来自 train / test，用于 direct 与 5-shot baseline。

## 当前桶规模

- `balanced_15x5_clean`: 15
- `balanced_15x5_rule1`: 15
- `balanced_15x5_rule2`: 15
- `balanced_15x5_rule3`: 15
- `balanced_15x5_rule4`: 15
- `balanced_15x5`: 75
