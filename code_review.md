# Code Review Checklist

## A. 架构与边界
- Point 1 是否仍然只依赖 common / benchmark / eval？
- Point 2 是否没有直接 import Point 1 的算法实现？
- 是否把研究逻辑与系统/服务逻辑分开？
- 是否避免了“临时脚本直接成为核心模块”？

## B. 可读性
- 文件职责是否单一？
- 命名是否表达真实语义，而不是临时缩写？
- 核心函数是否短小、可单测、输入输出清晰？
- 是否存在跨文件隐藏副作用？

## C. 协议正确性
- 数据 schema 是否显式定义？
- JSON 输出是否有测试覆盖？
- benchmark 字段名、坐标格式、split 口径是否与官方一致？
- Point 1 是否仍保持闭域四规则实验口径？

## D. 评测与复现
- 是否提供最小可运行命令？
- 是否更新了 configs / docs / tests？
- 指标实现是否与文档一致？
- 如果引入新依赖，是否必要且已记录？

## E. 研究有效性
- Point 1 是否真的输出了 evidence -> executor -> explanation 的证据链？
- Point 2 是否真的体现了 RAG / LoRA 协同，而不是简单换模型？
- 是否保留了错误分析、分层统计与失败案例导出接口？
