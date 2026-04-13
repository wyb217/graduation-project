# 12 Point 1 API Baseline

## 目标

在不引入 Point 1 主方法之前，先建立两个可运行的黑箱 API 对照组：

1. `direct` / zero-shot
2. `five_shot`

两者都输出同一套结构化 JSON，并共用同一套 benchmark reader。

## 为什么先做它

这样可以先回答：

- 数据是否能稳定送入多模态 API？
- 结构化 JSON 是否能稳定返回并解析？
- 5-shot 是否比 direct 更稳？
- 在不加我们方法的前提下，黑箱 VLM 的闭域能力有多强？

## 当前实现边界

本轮 baseline 只做：

- OpenAI-compatible provider 配置读取
- direct prompt
- 5-shot prompt
- 结构化 JSON parser
- subset 级运行脚本
- official-style prediction export bridge

本轮不做：

- 完整 official test 3004 跑分
- Point 1 evidence / executor 主方法
- RAG / LoRA / PEFT

## provider 策略

优先顺序：

1. `modelscope`
2. `dashscope`
3. 其他 OpenAI-compatible provider

`configs/system/providers.local.json` 为本地私有配置，不纳入版本控制。

## 子集设计

### `balanced_dev_15x5`

- 来源：train
- 用途：调 prompt、固定 5-shot 示例
- 规模：75

### `balanced_test_13x5`

- 来源：test
- 用途：快速评估
- 规模：65

说明：test 中纯净 `rule4` 单违规样本只有 13 张，因此 test 侧不能构造 15x5。

## 输出协议

baseline 统一返回：

```json
{
  "image_id": "0000424",
  "predictions": [
    {
      "rule_id": 1,
      "decision_state": "violation",
      "target_bbox": [0.1, 0.2, 0.3, 0.4],
      "supporting_evidence_ids": [],
      "counter_evidence_ids": [],
      "unknown_items": [],
      "reason_slots": {},
      "reason_text": "...",
      "confidence": 0.8
    }
  ]
}
```

## 5-shot 选择方式

默认从 `balanced_dev_15x5` 中固定取：

- clean 第 1 张
- rule1 第 1 张
- rule2 第 1 张
- rule3 第 1 张
- rule4 第 1 张

这样可以确保 few-shot 提示完全可复现。

## 当前 few-shot 形式

当前 5-shot 代码已改成更接近论文描述的 author-style VQA few-shot：

- 输入 5 个图像 + 5 个标注示例
- 示例答案使用：
  - `violated_rule_ids`
  - `explanation`
  - `target_bbox`
- 最终输出仍会被适配回本仓库统一的四规则结构化协议，便于评测

## eval bridge

当前仓库已经补上一个最小的 official eval bridge：

- 入口脚本：`scripts/run_point1_eval.py`
- 目标：把 baseline 输出 JSON 转成 ConstructionSite10k 官方风格预测文件
- 额外能力：可选同时生成本仓库内部 summary

示例命令：

```bash
python scripts/run_point1_eval.py \
  --baseline-output artifacts/point1/fiveshot-modelscope-balanced_test_13x5.json \
  --official-output artifacts/point1/fiveshot-modelscope-balanced_test_13x5.official.json \
  --registry src/benchmark/splits/constructionsite10k_balanced_test_13x5.json \
  --subset-name balanced_test_13x5 \
  --summary-output artifacts/point1/fiveshot-modelscope-balanced_test_13x5.eval-summary.json
```

bridge 的导出规则：

- `decision_state == "violation"` 的预测会映射到 `rule_{k}_violation`
- `target_bbox` 会落到官方格式的 `bounding_box`
- `reason_text` 会落到官方格式的 `reason`
- parse 失败样本会退化成四条规则全 `null` 的官方记录，避免丢样本
