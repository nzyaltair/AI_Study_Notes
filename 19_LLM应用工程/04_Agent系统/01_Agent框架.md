# Agent框架

## 背景与发展

随着[[大语言模型]]能力提升，构建Agent系统需要系统化的框架支持。从LangChain（2022）的模块化工具链，到AutoGen的多Agent对话，再到LangGraph的状态图工作流和OpenAI Agents SDK，Agent框架经历了从简单链式调用到复杂多Agent协作的演进。

## 核心思想

Agent框架的核心组成模块为：规划器（任务分解）、工具调用（外部能力）、记忆系统（上下文维持）、执行器（任务执行）和反思机制（自我改进）。不同框架在设计理念上各有侧重，但都围绕这五个核心模块展开。

## 技术原理

### Agent核心组成

| 模块 | 功能 | 关键技术 |
|------|------|----------|
| 规划器（Planner） | 任务分解为子任务序列 | [[思维链]]、思维树、规划验证 |
| 工具调用（Tool Use） | 调用外部API和工具 | Function Calling、动态工具注册 |
| 记忆系统（Memory） | 存储和检索历史经验 | [[向量数据库]]、记忆摘要、压缩 |
| 执行器（Executor） | 执行子任务，调用工具 | 同步/异步/并行执行、状态跟踪 |
| 反思机制（Reflection） | 反思行为，改进决策 | 自评估、错误检测、经验提取 |

### 规划范式对比

| 范式 | 核心思想 | 流程 | 适用场景 |
|------|----------|------|----------|
| [[ReAct]] | 推理与行动交替 | Thought→Action→Observation循环 | 通用Agent，工具调用频繁 |
| Plan-and-Execute | 先规划再执行 | 生成计划→依次执行→监控调整 | 结构化强的复杂任务 |
| Reflexion | 失败后反思重试 | 执行→评估→反思→重试 | 调试类，需要自我修正 |
| Multi-Agent Debate | 多Agent辩论 | 提出观点→反驳→综合 | 需要多角度思考的问题 |

### 主流框架对比

| 框架 | 核心定位 | 多Agent支持 | 工具调用 | 学习曲线 | 适用场景 |
|------|----------|-------------|----------|----------|----------|
| LangChain | 模块化工具链 | 中 | 强 | 高 | 通用LLM应用 |
| LangGraph | 状态图工作流 | 强 | 强 | 中 | 复杂决策流程 |
| AutoGen | 多Agent对话 | 强 | 中 | 中 | 多Agent协作 |
| CrewAI | 角色化协作 | 强 | 中 | 低 | 流程化团队任务 |
| OpenAI Agents SDK | 官方托管Agent | 中 | 强 | 低 | 快速部署 |
| LlamaIndex | RAG增强Agent | 弱 | 中 | 中 | 知识密集型应用 |
| DSPy | 程序合成 | 弱 | 中 | 高 | 精确控制，研究型 |

### 单Agent vs 多Agent架构选型

**单Agent适用**：任务单一、工具数量少（<10）、无需协作、延迟敏感。

**多Agent适用**：任务复杂需分工、需要多角度思考、各子任务需要不同专业能力、可容忍较高延迟。

### 多Agent协作模式

- **层层式**：主管Agent分配任务给子Agent（CrewAI Hierarchical）
- **对话式**：Agent间对话讨论（AutoGen GroupChat）
- **流水线式**：Agent串联各负责一步（研究→写作→审核）

## 发展演进

LangChain（2022，模块化工具链）→ AutoGPT（2023，自主Agent探索）→ AutoGen（2023，多Agent对话）→ LangGraph（2024，状态图工作流）→ CrewAI（2024，角色化协作）→ OpenAI Agents SDK（2024，官方框架）

## 关键算法·模型

- [[ReAct]]：推理与行动交替（Yao et al., 2022）
- Reflexion：反思机制（Shinn et al., 2023）
- Plan-and-Execute：先规划后执行
- CICERO：Meta的社交互动Agent架构
- MemGPT：操作系统式记忆管理（Packer et al., 2023）

## 应用场景

- 编程助手：Cursor、Claude Code、OpenDevin
- 数据分析Agent：自主查询、分析、可视化
- 内容生产流水线：研究→写作→审核
- 客服自动化：分类→检索→生成→回复
- 多Agent协作研究：文献检索、分析、综述

## 与其他技术关系

- Agent框架以[[大语言模型]]为推理核心
- [[工具调用]]与[[MCP]]提供工具集成能力
- [[RAG]]作为Agent的知识来源
- [[提示工程]]定义Agent行为
- [[结构化输出]]确保工具调用参数可解析
