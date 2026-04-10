# 09 Point 1 Foundation Status

## 本轮完成内容

已完成 Point 1 的第一阶段基础设施实现：

1. Python 项目骨架与 `src/` 布局；
2. `common.schemas.NormalizedBBox`；
3. `common.schemas.Point1Prediction` / `Point1Evidence`；
4. ConstructionSite10k typed sample / violation / attributes schema；
5. annotation parser；
6. dataset loader；
7. split registry 读取接口；
8. bbox / schema / parser / loader / registry 测试；
9. README 的环境与命令说明。

## 当前代码边界

- Point 1 仍未接入任何 VLM、外部 API、RAG、LoRA、QLoRA；
- Point 2 仅保留目录边界，不含实现；
- Point 3 仅保留系统层占位目录。

## 关键文件

- `src/common/schemas/bbox.py`
- `src/common/schemas/point1.py`
- `src/benchmark/constructionsite10k/types.py`
- `src/benchmark/constructionsite10k/parser.py`
- `src/benchmark/constructionsite10k/loader.py`
- `src/benchmark/constructionsite10k/registry.py`

## 下一阶段建议

1. 增加 Point 1 baseline 接口与结构化 JSON parser；
2. 增加 official evaluation bridge wrapper；
3. 先实现 Rule 1 的 candidate / predicate / executor 主链路；
4. 再进入 Rule 4 pair reasoning 与 edge-related modules。
