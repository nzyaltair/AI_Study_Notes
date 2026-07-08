# 方向十九：LLM应用工程

## 定义与范围

以LLM为推理核心，通过提示工程、检索增强生成（RAG）、Agent架构、工具调用和结构化输出构建端到端智能应用系统的工程方法论。

## 历史演进

| 年代 | 里程碑 | 意义 |
|------|--------|------|
| 2020 | GPT-3 Few-Shot | 提示工程开端 |
| 2020 | RAG（Lewis等） | 检索增强生成 |
| 2022 | CoT/ReAct | 推理引导与自主行动 |
| 2023 | Function Calling | LLM原生工具调用 |
| 2023 | LangChain/AutoGPT | Agent框架生态爆发 |
| 2024 | MCP协议 | 工具调用标准化 |
| 2024 | Computer Use Agent | GUI操作Agent |
| 2025 | Agent工程化 | Agent平台与生产运维 |

## 核心概念与方法

- **提示工程**：Zero/Few-Shot、Chain-of-Thought、ReAct、Self-Consistency、DSPy自动优化
- **RAG系统**：检索器+生成器、向量数据库、HyDE查询重写、混合检索、Cross-Encoder重排
- **Agent系统**：规划→行动→观察循环、记忆机制、多Agent协作、反思与自我修正
- **工具调用**：Function Calling、MCP协议、工具注册与编排
- **结构化输出**：JSON Schema约束、Constrained Decoding、Grammar-based Decoding

## 核心公式推导

### 1. RAG检索-生成范式

$$P(a|q) = \sum_{d \in \mathcal{R}} P(a|q, d) \cdot \text{sim}(q, d)$$

### 2. ReAct循环形式化

$$\text{Thought}_t \to \text{Action}_t \to \text{Observation}_t \to \text{Thought}_{t+1}$$

终止条件：$\text{Action} = \text{Finish}[\text{answer}]$

## 代表论文与里程碑

- Wei et al. (2022): Chain-of-Thought Prompting
- Yao et al. (2022): ReAct: Synergizing Reasoning and Acting
- Lewis et al. (2020): RAG for Knowledge-Intensive NLP Tasks
- Schick et al. (2023): Toolformer
- Anthropic (2024): Model Context Protocol（MCP）

## 学习资源

- LangChain / LlamaIndex / AutoGen 官方文档
- Hugging Face Agents Course
- DSPy: Programming Foundation Models

## 笔记要点与实践任务

- 实现CoT+Self-Consistency在数学推理上的对比
- 搭建RAG系统（向量检索+重排+生成）
- 实现ReAct Agent完成信息检索任务
- 对比不同Agent框架的架构与适用场景

## 与其他方向关系

- 「大语言模型」是其核心推理引擎
- 「RAG」「Agent」「提示工程」是其三大支柱
- Agent系统依赖「工具调用」与「MCP协议」
- 「模型部署」为LLM应用提供工程化基础设施
