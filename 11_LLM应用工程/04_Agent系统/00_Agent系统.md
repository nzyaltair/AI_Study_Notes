# Agent系统

## 背景与发展

Agent是[[大语言模型]]应用的高级形态——能够自主规划、使用工具、观察反馈并迭代执行的智能体。从ReAct（Yao et al., 2022）到AutoGPT（2023）再到现代Agent框架，Agent代表了AI从"回答问题"到"完成任务"的范式转变。

## 核心思想

Agent = LLM + 规划 + 工具 + 记忆 + 反馈循环。LLM作为"大脑"进行推理决策，工具扩展行动能力，记忆维持上下文，反馈循环实现自我修正。

## 技术原理

### Agent核心循环

```
Thought（推理）→ Action（行动/工具调用）→ Observation（观察结果）→ 循环 → Final Answer
```

### 规划（Planning）

| 方法 | 原理 | 适用 |
|------|------|------|
| [[ReAct]] | 交替推理与行动 | 通用Agent |
| Plan-and-Execute | 先制定完整计划再执行 | 复杂任务 |
| Tree of Thoughts | 树形搜索推理路径 | 探索性任务 |
| Reflexion | 失败后反思再重试 | 调试类任务 |

### 记忆（Memory）

| 类型 | 实现 | 作用 |
|------|------|------|
| 短期记忆 | 对话上下文窗口 | 当前任务上下文 |
| 长期记忆 | [[向量数据库]]存储 | 跨会话知识 |
| 工作记忆 | Scratchpad/笔记 | 中间推理结果 |
| 情景记忆 | 过去交互记录 | 个性化适配 |

### 工具使用

Agent通过Function Calling调用外部工具：搜索引擎、代码执行器、文件操作、API调用、数据库查询、[[RAG]]检索。

### 多Agent协作

| 模式 | 原理 | 示例 |
|------|------|------|
| 层层式 | 主管Agent分配任务给子Agent | 项目管理 |
| 对话式 | Agent间对话讨论 | 代码审查 |
| 竞争式 | 多Agent各自解决，取最优 | 数学求解 |
| 流水线式 | Agent串联，各负责一步 | 内容生产 |

## 发展演进

ReAct（2022）→ AutoGPT（2023，自主Agent热潮）→ Agent框架成熟（LangChain/AutoGen/CrewAI）→ 推理模型增强Agent规划（o1, 2024）→ Computer Use Agent（2024，GUI操作）

## 关键算法·模型

- [[ReAct]]：推理与行动交替（Yao et al., 2022）
- Reflexion：反思机制（Shinn et al., 2023）
- Tree of Thoughts：思维树搜索（Yao et al., 2023）
- Multi-Agent Debate：多Agent辩论
- CICERO：Meta的社交互动Agent

## 应用场景

- 编程助手（Cursor、Claude Code）
- 数据分析（自主查询数据库、生成报告）
- 研究助手（文献检索、综述生成）
- 自动化办公（邮件处理、日程管理）
- Web Agent（浏览器操作、网页信息提取）
- 多模态Agent（结合视觉能力的UI操作）

## 与其他技术关系

- Agent以[[大语言模型]]为核心推理引擎
- [[思维链]]和[[ReAct]]是Agent规划的基础
- [[工具调用]]与[[MCP]]是Agent的行动能力
- [[RAG]]是Agent的知识来源
- [[提示工程]]定义Agent行为
- [[多模态模型]]扩展Agent感知范围
