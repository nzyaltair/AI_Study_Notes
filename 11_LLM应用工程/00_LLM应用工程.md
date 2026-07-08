# LLM应用工程

## 背景与发展

大语言模型（LLM）从纯文本生成工具演进为应用系统核心，经历了三个阶段：提示工程探索期（2020-2022）、RAG与工具调用集成期（2023）、Agent自主执行期（2024至今）。[[大语言模型]]能力的突破使AI从"被动回答"走向"主动行动"，催生了提示工程、检索增强生成、Agent系统、工具调用等技术方向。

## 核心思想

LLM应用工程以LLM为推理核心，通过提示工程引导行为、RAG增强知识、工具调用扩展行动能力、Agent架构实现自主规划与执行，构建端到端的智能应用系统。

## 技术原理

### 核心范式

```
用户请求 → Agent规划（Thought）→ 选择工具（Action）→ 执行（Observation）→ 循环 → 最终回答
```

### 主要内容

- [[提示工程]]：提示词设计、Few-Shot、CoT、ReAct、提示模板与工程化
- [[结构化输出]]：JSON Schema约束、Constrained Decoding、Grammar-based Decoding
- [[RAG]]：检索增强生成、向量数据库、文档处理、检索优化、多模态RAG
- [[Agent]]：Agent架构（规划→行动→观察）、主流框架、记忆机制、多Agent协作
- [[MCP]]：Function Calling、Tool Use、Model Context Protocol协议

## 发展演进

提示工程（2020）→ RAG（2020，Lewis et al.）→ Function Calling（2023）→ Agent框架（2023，LangChain/AutoGen）→ MCP协议（2024，标准化工具连接）→ Computer Use Agent（2024，GUI操作）

## 关键算法·模型

- 提示工程：CoT、ReAct、Tree of Thoughts、Self-Consistency
- RAG：HyDE、多向量检索、倒数排名融合（RRF）、Cross-Encoder重排
- Agent：ReAct、Plan-and-Execute、Reflexion、多Agent辩论
- 结构化输出：Constrained Decoding、Grammar-based Decoding、Outlines

## 应用场景

- 智能问答系统与知识库
- 编程助手（Cursor、Claude Code）
- 数据分析与自动化报告
- 客服自动化与工单处理
- 多模态内容理解与生成

## 与其他技术关系

- Agent以[[大语言模型]]为核心推理引擎
- [[思维链]]推理是Agent规划的基础
- [[多模态模型]]扩展Agent的感知范围
- Agent部署依赖系统工程化与运维
- [[结构化输出]]确保Agent输出可解析
