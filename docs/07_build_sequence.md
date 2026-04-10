# 07 Build Sequence

## 原则
从零搭仓库时，不要让 Codex 一次性写完整系统。先冻结边界和契约，再分阶段实现。

## 推荐顺序
### Phase 0: 初始化仓库
- 建立目录树
- 建立 `pyproject.toml`
- 建立 lint / format / test 基础设施
- 建立最小 README、AGENTS、docs

### Phase 1: Benchmark 层
- 实现 ConstructionSite10k loader
- 实现 split registry
- 实现 annotation parser
- 实现最小可视化与 sanity checks

### Phase 2: 公共契约层
- 定义 bbox、sample、prediction、evidence、evaluation report schema
- 建立统一 serialization / deserialization

### Phase 3: Point 1 baseline
- direct VLM baseline 接口
- 结构化 JSON parser
- internal metrics
- official bridge wrapper

### Phase 4: Point 1 主方法
- Rule 1 -> Rule 4 -> Edge module -> Rule 2/3
- executor / explanation
- error analysis / ablations

### Phase 5: Point 2 baseline
- knowledge store format
- retrieval baseline
- prompt baseline
- LoRA dataset builder

### Phase 6: Point 2 主方法
- RAG-only
- LoRA-only
- LoRA+RAG
- robustness evaluation

### Phase 7: Point 3 scaffold
- 服务接口
- artifact 消费
- 人工复核入口
- 报告导出

## 对 Codex 的执行要求
每个 phase 结束时必须产出：
- 可运行命令
- 对应 tests
- 最小文档更新
- 变更摘要
