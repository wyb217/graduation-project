# 08 First Codex Prompt

建议把下面这段作为新仓库的第一次 Codex 指令：

```text
Read AGENTS.md and all docs/ files first.
Do not use the old repository as implementation context.
This is a from-scratch rebuild.

Your task for this turn:
1. Scaffold the repository exactly according to docs/02_repo_blueprint.md.
2. Keep Point 1 and Point 2 decoupled at the code level.
3. Do not implement models yet.
4. Create the minimal benchmark loader and schema contracts for ConstructionSite10k.
5. Add tests for loader, bbox schema, and split registry.
6. Add a concise README with setup, lint, format, and test commands.
7. Summarize the created files and remaining TODOs.

Non-negotiable constraints:
- Point 1 is closed-domain ConstructionSite10k only.
- No RAG / LoRA / external standards retrieval in Point 1.
- Code must be readable, typed, and modular.
- Do not create notebooks for core logic.
```

第二轮再让 Codex 做：
- direct baseline
- official bridge wrapper
- Point 1 Rule 1 path

不要一开始就让它“把整个项目做完”。
