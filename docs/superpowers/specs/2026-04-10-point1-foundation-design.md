# Point 1 Foundation Design

## 背景与目标

本轮只实现 Point 1 的**第一阶段基础设施**，不实现具体视觉模型与规则推理算法。

目标是把后续 Point 1 方法所依赖的基础层先固定下来，包括：

1. 仓库基础工程结构；
2. ConstructionSite10k benchmark 读取与解析；
3. 冻结的 split registry；
4. Point 1 / Point 2 可共享的公共 schema；
5. 最小可运行命令、测试与文档。

这样可以保证后续 Rule 1 / Rule 4 / Rule 2 / Rule 3 的实现建立在清晰、可测试、可复现的契约之上，而不是一开始就把 benchmark、方法、脚本、输出协议混在一起。

## 本轮明确范围

### 包含

- 建立 `pyproject.toml`，配置 `ruff`、`pytest` 与 `src/` 布局；
- 建立 `src/common` / `src/benchmark` / `src/eval` / `src/point1` / `src/point2` 的最小目录骨架；
- 定义基础 schema，例如：
  - `NormalizedBBox`
  - benchmark sample / annotation 契约
  - Point 1 结构化 prediction / evidence / explanation 契约
- 实现 ConstructionSite10k annotation parser；
- 实现 dataset loader；
- 实现 split registry 读写与冻结机制；
- 添加 loader、schema、registry 的单元测试；
- 更新 README，给出 setup / lint / format / test 命令。

### 不包含

- 外部 API 调用；
- VLM 推理；
- RAG / LoRA / PEFT；
- Point 1 具体规则执行器；
- Point 1 baseline 模型；
- Point 2 任何功能；
- Point 3 系统层。

## 设计原则

1. **闭域优先**：所有 Point 1 代码只服务于 ConstructionSite10k 四规则闭域口径。
2. **契约先行**：先固定 bbox、sample、prediction 等 I/O 契约，再接方法。
3. **强边界**：Point 1 不依赖 knowledge / finetune / system。
4. **可读可测**：每个模块职责单一，优先 dataclass / TypedDict / pydantic 中一种显式表达协议。
5. **脚本薄化**：`scripts/` 只做命令入口，不存放核心逻辑。

## 推荐实现方案

### 方案选择

本轮采用“**基础优先**”方案：

- 先完成 benchmark 层与公共契约层；
- 再在下一轮添加 Point 1 baseline 与 Rule 1 真实视觉路径。

### 为什么不直接做 Rule 1

如果直接做 Rule 1，很容易在没有固定 sample/schema/输出协议的情况下，把候选生成、标注解析、输出 JSON、评测输入全部耦合在同一条流水线里，后续返工成本高。

### 为什么这一步足够有价值

完成本轮后，仓库将具备以下能力：

- 稳定读取 ConstructionSite10k annotation；
- 冻结可复现实验 split；
- 为 Point 1 和 Point 2 提供共享协议；
- 为 baseline、rule executor、official bridge 提供稳定输入输出。

## 模块划分

### `src/common`

职责：通用协议与无业务依赖的工具。

首轮建议包含：

- `src/common/schemas/bbox.py`：归一化 bbox 契约与校验；
- `src/common/schemas/point1.py`：Point 1 prediction / evidence / explanation 契约；
- `src/common/io/json.py`：统一 JSON 读写；

约束：

- 不依赖 `benchmark`、`point1`、`point2`。

### `src/benchmark/constructionsite10k`

职责：ConstructionSite10k 的 benchmark 读取与字段语义表达。

首轮建议包含：

- `types.py`：annotation/sample 数据结构；
- `parser.py`：原始 annotation → 强类型对象；
- `loader.py`：按 split / image_id 加载样本；
- `registry.py`：冻结 split registry；

约束：

- 只负责 benchmark 语义与数据访问，不包含 Point 1 算法逻辑。

### `src/eval`

职责：本轮只预留目录，不实现复杂指标。

可先提供：

- `src/eval/reports/` 占位；
- 后续接 official bridge 与内部指标时再扩展。

### `src/point1`

职责：本轮只建立边界清晰的空目录与公共入口，不实现规则逻辑。

首轮只保留：

- `pipelines/` 占位；
- 未来 baseline 与 executor 从这里进入。

### `src/point2`

职责：本轮只建立空骨架，确保目录存在但不与 Point 1 混写。

## 数据流设计

本轮数据流固定为：

`raw annotation json -> parser -> typed sample -> split loader -> downstream consumer`

关键约束：

- bbox 统一为 normalized `xyxy`；
- 原始字段名不在下游散落解析，统一在 parser 里收口；
- split 不通过脚本参数临时拼接，而是读取冻结 registry；
- Point 1 结构化输出 schema 先定义好，但本轮不产生真实预测。

## 错误处理设计

- 对缺失关键字段、非法 bbox、未知 rule 字段提供显式异常；
- 对可选字段（例如某些类别为空列表或某条违规为 `null`）做宽容解析；
- 所有解析错误都应尽量带上 `image_id` 或原始字段上下文，方便后续清洗数据。

## 测试策略

本轮测试只覆盖基础层，重点是“口径正确性”而不是模型表现。

### 必测项

1. `NormalizedBBox`
   - 合法范围；
   - 非法坐标报错；
   - 坐标顺序校验。

2. annotation parser
   - 能解析官方风格样例；
   - `rule_n_violation = null` 正常处理；
   - list 类 object field 正常处理。

3. split registry
   - 能从 registry 读取固定 split；
   - 未知 split 报错；
   - 顺序与内容稳定。

4. dataset loader
   - 能按 image_id / split 返回样本；
   - 能把 parser 与 registry 串起来。

## 可运行命令

本轮结束后应至少支持：

- `ruff check .`
- `ruff format .`
- `pytest -q`

如果提供示例脚本，可增加：

- `python -m scripts.inspect_cs10k_sample --help`

但脚本不是本轮重点。

## 第二轮预期衔接

本轮完成后，下一轮直接进入：

1. Point 1 baseline 接口；
2. 结构化 JSON parser；
3. Rule 1 的 candidate / predicate / executor 路径；
4. official bridge wrapper。

## 风险与规避

### 风险 1：过早抽象

规避：本轮只做 benchmark 和 schema 的必要最小实现，不设计复杂插件系统。

### 风险 2：把 Point 1 输出协议写死在模型实现里

规避：先在 `src/common/schemas/point1.py` 固定结构化输出契约。

### 风险 3：split 管理散落在脚本参数中

规避：使用 registry 文件冻结 subset / dev / test。

## 验收标准

以下条件满足即可认为本轮完成：

1. 仓库具备最小 Python 工程结构；
2. ConstructionSite10k loader / parser / registry 可运行；
3. bbox / sample / point1 output schema 已显式定义；
4. 测试通过；
5. README 与相关文档已更新；
6. Point 1 与 Point 2 目录边界清楚，没有实现层耦合。
