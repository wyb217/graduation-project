# 06 External References

## 需要固定给 Codex 的外部参考
以下链接建议写进仓库文档，并在真正开工前固定版本（最好补 commit/tag/访问说明）。

### A. ConstructionSite10k 数据与论文
1. Dataset card
   https://huggingface.co/datasets/LouisChen15/ConstructionSite
2. Paper
   https://arxiv.org/abs/2508.11011
3. Official implementation / evaluation repo
   https://github.com/LouisChen15/ConstructionSite-10k-Implementation

### B. 你自己的研究上下文（建议整理成 repo 内文档，而不是原样丢 PDF）
1. 开题收敛方案
2. Point 1 研究计划
3. Point 1 TODO phase list
4. 研究现状综述
5. NetSI 开题流程约束

## 是否必须给 Codex 官方 implement repo 地址？
结论：**建议给，而且应当给。**

但用途必须被限定为：
- benchmark 语义参考
- 评测脚本与 bridge 参考
- 示例 annotation / inference 输入输出参考

不要把它当成：
- 新仓库骨架
- 直接继承的代码风格来源
- 你的主 pipeline 模板

## 最佳实践
- 把这些 URL 放在 repo 内文档，而不是每次临时提示里重复粘贴。
- 对动态外部资源，尽量补 commit、tag、发布日期或访问日期。
- 如果某个外部仓库只取用一个目录或一个脚本，要在文档里点名具体用途。
