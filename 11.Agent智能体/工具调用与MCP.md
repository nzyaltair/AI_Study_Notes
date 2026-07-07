# 工具调用与MCP

## 背景

工具调用（Function Calling/Tool Use）使LLM能调用外部API和工具，是从"聊天机器人"到"[[Agent系统|Agent]]"的关键能力。MCP（Model Context Protocol）是Anthropic提出的开放标准，旨在统一LLM与外部工具/数据源的连接方式。

## 核心思想

LLM不直接执行操作，而是根据用户意图选择合适的工具并生成结构化调用参数，由外部执行器执行后返回结果，LLM再基于结果继续推理。

## 技术原理

### Function Calling流程

```
1. 用户请求 → LLM分析意图
2. LLM选择工具，生成调用参数（JSON）
3. 执行器调用实际API
4. 返回结果给LLM
5. LLM基于结果生成最终回答
```

### OpenAI Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京天气如何？"}],
    tools=tools
)

# LLM返回工具调用
tool_call = response.choices[0].message.tool_calls[0]
# 执行: get_weather(city="北京")
# 将结果返回给LLM继续生成
```

### 并行工具调用

现代LLM支持单次返回多个工具调用，并行执行以提升效率：

```
LLM → [get_weather("北京"), get_news("科技"), search_db("用户数据")]
     → 并行执行 → 汇总结果 → LLM生成回答
```

### MCP（Model Context Protocol）

Anthropic提出的开放标准，统一LLM与外部资源的连接：

```
LLM应用 (Client) ←→ MCP协议 ←→ MCP Server ←→ 数据源/工具
```

| MCP概念 | 作用 |
|---------|------|
| **MCP Server** | 暴露工具和数据给LLM |
| **MCP Client** | LLM应用中的MCP连接器 |
| **Resources** | 可读取的数据源（文件、数据库） |
| **Tools** | 可调用的函数/API |
| **Prompts** | 预定义的提示模板 |

**MCP优势**：
- 标准化：一个协议连接所有工具
- 解耦：工具开发与LLM应用独立
- 生态：社区共享MCP Server

### 工具定义最佳实践

- **描述清晰**：LLM根据description选择工具
- **参数schema严格**：用JSON Schema约束参数类型
- **错误处理**：工具执行失败时返回有用错误信息
- **工具数量**：过多工具会降低选择准确率
- **工具命名**：语义化命名帮助LLM理解

### 结构化输出

工具调用依赖LLM输出结构化数据：

| 方法 | 原理 | 可靠性 |
|------|------|--------|
| **Function Calling** | 模型原生支持 | 高 |
| **JSON Mode** | 强制输出合法JSON | 中高 |
| **Schema约束** | response_format指定schema | 高 |
| **Prompt引导** | 提示词引导输出JSON | 中 |
| **后处理解析** | 正则/解析提取 | 低 |

## 发展演进

API调用（硬编码）→ Function Calling（2023，OpenAI）→ Tool Use（通用化）→ MCP协议（2024，标准化）→ Computer Use（2024，GUI操作）

## 应用领域

- **[[Agent系统|Agent]]**：工具调用是Agent的核心能力
- **[[RAG系统|RAG]]**：RAG检索可作为工具调用
- **API编排**：LLM根据需求选择和组合API
- **代码执行**：生成并执行代码验证
- **数据库查询**：自然语言转SQL并执行
- **自动化工作流**：多工具组合完成复杂任务

## 与其他技术关系

- 工具调用是[[Agent系统|Agent]]的行动能力
- [[结构化输出|结构化输出]]是工具调用的基础
- MCP标准化了[[Agent智能体|Agent]]与工具的连接
- [[提示工程|提示工程]]影响工具选择的准确性
- [[LLM推理与优化|推理优化]]影响工具调用的延迟
