# 方向十九：LLM应用工程

## 定义与范围

以大语言模型为推理核心，通过提示工程、检索增强生成（RAG）、Agent架构、工具调用和结构化输出构建端到端智能应用系统的工程方法论。

## 历史演进

| 年代 | 里程碑 | 意义 |
|------|--------|------|
| 2020 | GPT-3 Few-Shot / RAG（Lewis等） | 提示工程开端 + 检索增强生成 |
| 2022 | CoT（Wei等）/ ReAct（Yao等） | 推理引导与自主行动 |
| 2023 | Function Calling / LangChain生态爆发 | LLM原生工具调用 + Agent框架 |
| 2023 | Toolformer / DSPy / Self-RAG | 自学习工具 + 提示优化 + 自适应检索 |
| 2024 | MCP协议 / LangGraph / Computer Use | 标准化工具连接 + 状态图工作流 + GUI操作 |
| 2025 | 推理模型驱动Agent / Agent平台化 | Agent工程化走向成熟 |

## 核心技术栈

- **提示工程**：Zero/Few-Shot、CoT、ReAct、Self-Consistency、ToT、DSPy自动优化
- **结构化输出**：JSON Schema约束、Constrained Decoding、Grammar-based Decoding
- **RAG系统**：检索器+生成器、向量数据库、HyDE查询重写、混合检索（RRF）、Cross-Encoder重排、Self-RAG/CRAG
- **Agent系统**：规划→行动→观察循环、记忆机制（短期/长期/工作记忆）、多Agent协作、反思与自我修正
- **工具调用**：Function Calling、MCP协议、工具注册与编排、Computer Use

## 核心公式

### RAG检索-生成范式

$$P(a|q) = \sum_{d \in \mathcal{R}} P(a|q, d) \cdot P(d|q)$$

### ReAct循环形式化

$$\text{Thought}_t \to \text{Action}_t \to \text{Observation}_t \to \text{Thought}_{t+1}$$

终止条件：$\text{Action} = \text{Finish}[\text{answer}]$

## 代表论文

- Wei et al. (2022): Chain-of-Thought Prompting
- Yao et al. (2022): ReAct: Synergizing Reasoning and Acting
- Lewis et al. (2020): RAG for Knowledge-Intensive NLP Tasks
- Schick et al. (2023): Toolformer
- Asai et al. (2023): Self-RAG
- Anthropic (2024): Model Context Protocol（MCP）

## 主流框架

LangChain / LangGraph / LlamaIndex / AutoGen / CrewAI / DSPy / OpenAI Agents SDK

## 笔记结构

| 笔记 | 主题 |
|------|------|
| [[LLM应用工程]] | 总体概述 |
| [[提示工程]] | 提示设计、CoT、ReAct、工程化 |
| [[结构化输出]] | JSON Schema、Constrained Decoding |
| [[RAG系统]] | 检索增强生成（基础/进阶/多模态/向量数据库） |
| [[Agent系统]] | Agent架构、框架、记忆、工作流 |
| [[工具调用与MCP协议]] | Function Calling、MCP、Computer Use |
| [[自动化工作流]] | 编排框架、多Agent协作、容错与可观测性 |
| [[LLM应用评估与可观测性]] | 评估方法、监控、追踪 |
| [[LLM应用安全]] | Prompt注入、数据安全、防护策略 |

## 与其他方向关系

- 「大语言模型」是其核心推理引擎
- 「RAG」「Agent」「提示工程」是其三大支柱
- Agent系统依赖「工具调用」与「MCP协议」
- 「模型部署」为LLM应用提供工程化基础设施
- 「量化技术」与「KV-Cache」直接影响应用性能与成本
