# 02 Repo Blueprint

## 推荐仓库形态
建议采用 **单仓库 + 强边界 monorepo**，而不是把三个点混成一个 `src/` 大目录，也不是完全拆成三个零散仓库。

推荐原因：
- benchmark、schema、评测工具、可视化工具可以共享。
- Point 1 / Point 2 / Point 3 仍可通过目录与依赖规则强制隔离。
- 论文与工程产物集中，管理更稳定。

## 推荐目录树
```text
repo/
├─ AGENTS.md
├─ README.md
├─ code_review.md
├─ pyproject.toml
├─ configs/
│  ├─ point1/
│  ├─ point2/
│  └─ system/
├─ docs/
├─ scripts/
│  ├─ download_cs10k.py
│  ├─ run_point1_baseline.py
│  ├─ run_point1_eval.py
│  ├─ run_point2_baseline.py
│  ├─ run_point2_eval.py
│  └─ export_artifacts.py
├─ src/
│  ├─ common/
│  │  ├─ io/
│  │  ├─ schemas/
│  │  ├─ utils/
│  │  └─ viz/
│  ├─ benchmark/
│  │  ├─ constructionsite10k/
│  │  └─ splits/
│  ├─ eval/
│  │  ├─ metrics/
│  │  ├─ bridges/
│  │  └─ reports/
│  ├─ point1/
│  │  ├─ candidates/
│  │  ├─ predicates/
│  │  ├─ executor/
│  │  ├─ explanation/
│  │  └─ pipelines/
│  ├─ knowledge/
│  │  ├─ rules/
│  │  ├─ retrievers/
│  │  └─ casebase/
│  ├─ finetune/
│  │  ├─ datasets/
│  │  ├─ lora/
│  │  └─ trainers/
│  ├─ point2/
│  │  ├─ rag/
│  │  ├─ inference/
│  │  ├─ experiments/
│  │  └─ pipelines/
│  └─ system/
│     ├─ services/
│     ├─ api/
│     └─ ui/
├─ tests/
│  ├─ common/
│  ├─ benchmark/
│  ├─ point1/
│  ├─ point2/
│  └─ system/
├─ .agents/
│  └─ skills/
└─ artifacts/
   └─ .gitkeep
```

## 依赖规则
- `src/common` 不依赖业务层。
- `src/benchmark` 只负责数据与 split 契约。
- `src/eval` 只负责指标、评测桥接、报告导出。
- `src/point1` 不依赖 `src/knowledge`、`src/finetune`、`src/point2`、`src/system`。
- `src/point2` 不依赖 `src/point1` 算法实现。
- `src/system` 只能依赖稳定对外接口，不直接访问内部实验细节。

## 推荐技术栈
- Python 3.11+
- `ruff`：lint + format
- `pytest`：测试
- `pydantic` 或 dataclass：协议/配置对象
- `uv` 或标准 venv：环境管理（二选一，开仓时一次性定死）

## 不推荐的做法
- Point 1 和 Point 2 共用同一个 `pipeline.py`
- `notebooks/` 内部直接堆所有核心逻辑
- `scripts/` 里出现和 `src/` 重复的核心算法实现
- 所有结果都只保存在 notebook 输出单元而不落盘
