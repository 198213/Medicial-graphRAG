# 医药知识图谱与 GraphRAG 智能问答系统

基于 OpenKG 公开医疗数据集构建医药知识图谱，实现 GraphRAG 智能问答，支持自然语言查询疾病症状、推荐药物、推荐食物等场景。

## 项目亮点

- 23,111 实体、154,444 关系**的医药知识图谱，基于 Neo4j 存储
- GraphRAG Pipeline：LLM 意图识别 → LLM 实体抽取 → Cypher 模板检索 → LLM 受控生成
- 缓解幻觉：不让 LLM 自由写 Cypher，用意图分类 + 参数抽取 + 模板路由，保证查询确定性
- 可评估：15 条人工标注 QA，命中率 86.67%，空答案率 0%

## 架构

```mermaid
flowchart LR
  U[用户] --> S[graphrag.py]
  S --> I[LLM 意图识别]
  I --> E[LLM 实体抽取]
  E --> C[Cypher 模板路由]
  C --> N[(Neo4j 医药知识图谱)]
  N --> R[图谱结果 JSON]
  R --> G[LLM 受控生成]
  G --> U