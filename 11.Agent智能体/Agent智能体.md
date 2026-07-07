# Agent智能体

> Agent智能体主题索引，涵盖提示工程、RAG系统、Agent架构和工具调用。

## 概述

Agent是LLM从"对话工具"进化为"自主智能体"的关键。通过提示工程引导LLM行为，RAG增强知识获取，工具调用扩展行动能力，Agent能够自主规划、决策和执行复杂任务。这标志着AI从"被动回答"走向"主动行动"。

## 主要内容

### [提示工程](提示工程.md)

提示词设计、Few-Shot、CoT、ReAct、提示模板与工程化。

### [RAG系统](RAG系统.md)

检索增强生成、向量数据库、文档处理、检索优化、多模态RAG。

### [Agent系统](Agent系统.md)

Agent架构（规划→行动→观察）、主流框架、记忆机制、多Agent协作。

### [工具调用与MCP](工具调用与MCP.md)

Function Calling、Tool Use、Model Context Protocol（MCP）协议。

## 核心范式

```
用户请求 → Agent规划（Thought）→ 选择工具（Action）→ 执行（Observation）→ 循环 → 最终回答
```

## 与其他技术关系

- Agent以[[大语言模型|LLM]]为核心推理引擎
- [[推理模型与思维链|CoT推理]]是Agent规划的基础
- [[多模态模型|多模态能力]]扩展Agent的感知范围
- Agent部署依赖[[系统工程化与运维|系统工程化]]
- [[结构化输出]]确保Agent输出可解析
