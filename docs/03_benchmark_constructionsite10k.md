# 03 Benchmark: ConstructionSite10k

## 为什么它是 Point 1 的唯一主 benchmark
ConstructionSite10k 提供了 Point 1 所需的完整闭域要素：
- 图像级 caption
- 四条规则违规 VQA
- 违规 reason
- violator bounding box
- visual grounding 目标
- 场景属性：illumination / camera_distance / view / quality_of_info

这使 Point 1 可以在同一个 benchmark 下完成：
- 分类
- 定位
- 解释
- 分层误差分析

## 数据规模
- 总图像数：10,013
- Train：7,009
- Test：3,004

## 四条内置规则
1. Basic PPE
2. Safety harness when working at height without protection
3. Edge protection / warning for underground projects
4. Workers inside excavator blind spots or operating radius

## 官方样例 annotation 关键字段
```json
{
  "image_id": "0000424",
  "image_caption": "...",
  "illumination": "normal lighting",
  "camera_distance": "mid distance",
  "view": "elevation view",
  "quality_of_info": "rich info",
  "rule_1_violation": {
    "bounding_box": [[0.22, 0.59, 0.28, 0.75]],
    "reason": "..."
  },
  "rule_2_violation": null,
  "rule_3_violation": null,
  "rule_4_violation": null,
  "excavator": [[...]],
  "rebar": [],
  "worker_with_white_hard_hat": [[...]]
}
```

## 对 Codex 的明确要求
- 新仓库必须首先复现 dataset loader、annotation parser、split registry。
- 坐标统一采用 normalized `xyxy`；如需转换，必须集中在一个模块内。
- 任何 Point 1 实验都不得改写 benchmark 原始语义。
- balanced subset / dev subset / test subset 必须以 registry 文件冻结，不允许散落在脚本参数中。

## 如何使用官方实现仓库
官方实现仓库只用来做三件事：
1. 确认字段语义与示例格式。
2. 参考官方评测桥接与 notebook 说明。
3. 对齐 inference/evaluation 的输入输出格式。

不应用来做的事：
- 不作为新仓库目录模板。
- 不继承其临时 notebook 风格。
- 不直接沿用其特定模型 inference 脚本作为主 pipeline。
