# ConstructionSite10k 数据集统计摘要

本文件汇总了本仓库根目录下本地数据文件的三类统计结果：
1. 属性标签的样本分布
2. 规则与属性的交叉分布
3. violator bbox 的尺度统计

## 数据来源

- `train-00001-of-00002.parquet`
- `train-00002-of-00002.parquet`
- `test.parquet`
- 样本规模：train = **7009**, test = **3004**

## 1. 属性标签的样本分布

字段映射说明：
- `illumination`: `normal lighting / underexposed / overexposed / night`
- `camera_distance`: `short distance / mid distance / long distance`
- `view`: `plan view / elevation view`，下表映射为 `yes / no`
- `quality_of_info`: `poor info / rich info`，下表映射为 `sparse / rich`

| 属性 | 各取值 | train 数量 | test 数量 |
| --- | --- | --- | --- |
| illumination | normal / underexposed / overexposed / night | 5885 / 692 / 273 / 159 | 2426 / 381 / 154 / 43 |
| camera distance | short / mid / long | 1560 / 5063 / 386 | 1360 / 1309 / 334 |
| plan view | yes / no | 723 / 6286 | 109 / 2895 |
| quality of information | sparse / rich | 3029 / 3980 | 1713 / 1291 |

> 注：test split 中 `camera_distance` 有 **1** 条缺失值，因此该属性三类计数之和为 3003。

## 2. 规则与属性的交叉分布

统计口径：将 train 与 test 合并；一张图像只要对应 `rule_i_violation != null`，就记为该规则的一个正样本。
同一张图像可能同时命中多个规则，因此不同行之间不能直接相加。

| 规则 | 正样本总数（train/test） | normal | underexposed/night | overexposed | short/mid | long | sparse | rich |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rule 1 PPE | 1001（677/324） | 743 (74.2%) | 222 (22.2%) | 36 (3.6%) | 962 (96.1%) | 39 (3.9%) | 397 (39.7%) | 604 (60.3%) |
| Rule 2 harness | 84（59/25） | 63 (75.0%) | 15 (17.9%) | 6 (7.1%) | 74 (88.1%) | 10 (11.9%) | 34 (40.5%) | 50 (59.5%) |
| Rule 3 edge protection | 172（109/63） | 138 (80.2%) | 24 (14.0%) | 10 (5.8%) | 163 (94.8%) | 9 (5.2%) | 59 (34.3%) | 113 (65.7%) |
| Rule 4 excavator blind spot | 70（46/24） | 59 (84.3%) | 8 (11.4%) | 3 (4.3%) | 69 (98.6%) | 1 (1.4%) | 42 (60.0%) | 28 (40.0%) |

补充：`plan view` 交叉分布

| 规则 | plan view = yes | plan view = no |
| --- | --- | --- |
| Rule 1 PPE | 32 (3.2%) | 969 (96.8%) |
| Rule 2 harness | 3 (3.6%) | 81 (96.4%) |
| Rule 3 edge protection | 4 (2.3%) | 168 (97.7%) |
| Rule 4 excavator blind spot | 2 (2.9%) | 68 (97.1%) |

## 3. violator bbox 的尺度统计

统计口径：将 train 与 test 合并，使用数据集中的 normalized `xyxy` 坐标，按单个 violator bbox 实例统计。
面积占比定义为：`(x2 - x1) * (y2 - y1)`，即 bbox 占整张图像面积的比例。

清洗说明（未纳入有效 bbox 统计）：
- Rule 1 PPE: nonpositive_bbox_area=1, empty_bbox_list=2

| 规则 | 有效 bbox 数 | bbox面积均值 | bbox面积中位数 |
| --- | --- | --- | --- |
| Rule 1 PPE | 1661 | 3.87% | 1.36% |
| Rule 2 harness | 146 | 1.72% | 0.84% |
| Rule 3 edge protection | 199 | 20.04% | 16.38% |
| Rule 4 excavator blind spot | 71 | 1.89% | 1.02% |

| 规则 | camera distance | 有效 bbox 数 | bbox面积均值 | bbox面积中位数 |
| --- | --- | --- | --- | --- |
| Rule 1 PPE | short | 523 | 8.43% | 5.22% |
| Rule 1 PPE | mid | 1082 | 1.83% | 0.96% |
| Rule 1 PPE | long | 56 | 0.76% | 0.49% |
| Rule 2 harness | short | 14 | 4.84% | 4.89% |
| Rule 2 harness | mid | 105 | 1.53% | 0.80% |
| Rule 2 harness | long | 27 | 0.83% | 0.32% |
| Rule 3 edge protection | short | 27 | 23.96% | 20.16% |
| Rule 3 edge protection | mid | 161 | 19.54% | 16.49% |
| Rule 3 edge protection | long | 11 | 17.82% | 15.05% |
| Rule 4 excavator blind spot | short | 25 | 3.23% | 1.92% |
| Rule 4 excavator blind spot | mid | 45 | 1.17% | 0.78% |
| Rule 4 excavator blind spot | long | 1 | 0.65% | 0.65% |

## 简要观察

- 四条规则的正样本都高度集中在 `short/mid distance`。
- `Rule 3 edge protection` 的目标框显著更大，说明它更接近大区域风险目标。
- `Rule 1 / Rule 2 / Rule 4` 更接近小目标，且 camera distance 越远，bbox 面积越小。
- `Rule 4 excavator blind spot` 更偏向 `sparse` 信息质量。

## 复现说明

- 本文档统计均直接基于本地 parquet 文件计算得到。
- 若后续需要论文排版，可将本文件中的 Markdown 表格进一步转换为 LaTeX 表格。
