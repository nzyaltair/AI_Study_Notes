# 分词与Tokenization

## 目录

1. [分词在LLM中的作用与位置](#分词在LLM中的作用与位置)
2. [分词方法的发展脉络](#分词方法的发展脉络)
3. [主流分词算法详解](#主流分词算法详解)
4. [常见Tokenzier对比与分析](#常见Tokenzier对比与分析)
5. [典型开源模型的Tokenizer配置](#典型开源模型的Tokenizer配置)
6. [Tokenizer的实际应用示例](#Tokenizer的实际应用示例)
7. [常见问题与解决方案](#常见问题与解决方案)
8. [Tokenization的局限性与挑战](#Tokenization的局限性与挑战)
9. [新兴方向与未来趋势](#新兴方向与未来趋势)
10. [总结与学习资源](#总结与学习资源)

## 1. 分词在LLM中的作用与位置

### 1.1 什么是分词与Tokenization

分词（Tokenization）是将连续的文本序列转换为离散的词汇单元（Token）的过程，是自然语言处理（NLP）和大型语言模型（LLM）中的基础前置步骤。

### 1.2 在LLM架构中的位置

在LLM的整体架构中，分词器（Tokenizer）位于最前端，负责：
- 将原始文本转换为模型可处理的数字序列
- 连接自然语言与模型内部表示
- 影响模型的[[词汇表]]大小、训练效率和生成质量

### 1.3 核心作用

1. **词汇表构建**：确定模型可处理的基本单元集合
2. **OOV（Out-of-Vocabulary）处理**：解决未登录词问题
3. **序列长度控制**：影响模型的上下文窗口大小
4. **多语言支持**：决定模型对不同语言的处理能力
5. **推理效率**：影响模型的计算速度和内存占用

## 2. 分词方法的发展脉络

### 2.1 基于规则的方法

- **空格/标点分割**：最基础的分词方法，适用于英语等以空格分隔的语言
- **字典匹配**：基于预定义字典进行最长匹配或正向/反向最大匹配
- **特点**：简单高效，但处理未登录词和歧义问题能力弱

### 2.2 按粒度分类的分词方法

| 粒度 | 描述 | 优点 | 缺点 | 典型应用 |
|------|------|------|------|----------|
| 字符级 | 将文本拆分为单个字符 | 词汇表小，无OOV问题 | 语义信息少，序列长度长 | 早期RNN模型 |
| 词级 | 将文本拆分为完整词汇 | 语义信息丰富，序列长度适中 | 词汇表大，OOV问题严重 | 传统NLP模型 |
| 子词级 | 将文本拆分为介于字符和词之间的单元 | 平衡词汇表大小和语义信息 | 实现复杂度较高 | 现代LLM |

### 2.3 子词级分词的兴起

子词级分词解决了词级分词的OOV问题和字符级分词的语义信息不足问题，成为现代LLM的主流选择。

## 3. 主流分词算法详解

### 3.1 Byte Pair Encoding (BPE)

#### 3.1.1 核心思想

BPE（字节对编码）是一种数据压缩算法，用于从训练语料中自动学习子词词汇表。

#### 3.1.2 算法步骤

1. **初始化**：将词汇表初始化为所有单个字符
2. **统计频率**：统计语料中所有相邻字符对的出现频率
3. **合并操作**：将出现频率最高的字符对合并为新的子词
4. **迭代执行**：重复步骤2-3，直到达到预设的词汇表大小

#### 3.1.3 示例过程

```
初始状态：h e l l o
合并最高频对 'll' → h e ll o
合并最高频对 'he' → he ll o
合并最高频对 'hello' → hello
```

#### 3.1.4 特点与应用

- 自底向上的合并策略
- 词汇表大小可控制
- 广泛应用于GPT系列、[[LLaMA]]等模型

### 3.2 WordPiece

#### 3.2.1 核心思想

WordPiece是BPE的变体，通过最大化语言模型的似然概率来选择要合并的子词对。

#### 3.2.2 算法步骤

1. **初始化**：与BPE相同，初始化为单个字符
2. **计算增益**：计算每个相邻字符对合并后的语言模型似然增益
3. **合并决策**：合并增益最大且大于阈值的字符对
4. **迭代执行**：直到达到预设词汇表大小

#### 3.2.3 特点与应用

- 更注重子词的语义合理性
- 广泛应用于BERT、[[ALBERT]]等模型

### 3.3 Unigram Language Model

#### 3.3.1 核心思想

Unigram语言模型是一种自顶向下的分词方法，从大词汇表开始，逐步移除贡献较小的子词。

#### 3.3.2 算法步骤

1. **初始化**：构建一个包含大量子词的初始词汇表
2. **计算概率**：训练unigram语言模型，计算每个子词的概率
3. **移除子词**：移除对语言模型似然贡献最小的子词
4. **重新评估**：重新计算剩余子词的概率
5. **迭代执行**：直到词汇表大小达到预设值

#### 3.3.3 特点与应用

- 能够生成更优的分词结果
- 支持多种分词路径
- 应用于[[XLM-RoBERTa]]、T5等模型

### 3.4 SentencePiece

#### 3.4.1 核心思想

SentencePiece是一种无监督的分词框架，将文本视为原始字节序列，支持BPE和Unigram两种算法。

#### 3.4.2 关键特性

- **字节级别处理**：支持任意语言，无需预先分词
- **可逆转换**：可以从Token序列完全恢复原始文本
- **统一处理**：将空格视为普通字符，使用`<unk>`标记表示
- **开放源代码**：提供跨平台的实现

#### 3.4.3 应用

广泛应用于多语言模型和跨语言迁移学习任务

### 3.5 TikToken

#### 3.5.1 核心思想

[[TikToken]]是OpenAI开发的高效BPE实现，用于GPT系列模型的分词。

#### 3.5.2 关键特性

- **高性能**：采用C++实现，编码速度快
- **字节级处理**：支持多种编码格式
- **灵活配置**：支持不同模型的词汇表

#### 3.5.3 应用

- GPT-3.5、GPT-4等OpenAI模型
- 高效的文本编码工具

## 4. 常见Tokenzier对比与分析

### 4.1 性能对比

| 分词算法 | 词汇表大小控制 | OOV处理能力 | 编码速度 | 多语言支持 | 可逆性 |
|----------|----------------|-------------|----------|------------|--------|
| BPE | 好 | 中 | 快 | 中 | 中 |
| WordPiece | 好 | 中 | 中 | 中 | 中 |
| Unigram | 好 | 好 | 慢 | 好 | 中 |
| SentencePiece | 好 | 好 | 中 | 好 | 好 |
| TikToken | 好 | 中 | 快 | 中 | 中 |

### 4.2 适用场景分析

- **BPE**：适用于单语言或主要语言模型
- **WordPiece**：适用于需要细粒度控制的模型
- **Unigram**：适用于多语言模型
- **SentencePiece**：适用于跨语言迁移学习
- **TikToken**：适用于需要高性能编码的场景

## 5. 典型开源模型的Tokenizer配置

### 5.1 GPT系列

- **GPT-3/3.5/4**：使用TikToken实现的BPE分词
  - 词汇表大小：50,257
  - 特殊token：`<|endoftext|>`
  - 编码方式：字节级BPE

### 5.2 LLaMA系列

- **LLaMA/LLaMA 2**：使用字节级BPE分词
  - 词汇表大小：32,000
  - 特殊token：`<s>`（开始）、`</s>`（结束）
  - 特点：支持多语言，中文处理能力有限

### 5.3 Qwen系列

- **Qwen 1.5**：使用字节级BPE分词
  - 词汇表大小：151,936
  - 特点：优化了中文处理，支持多语言

### 5.4 ChatGLM系列

- **ChatGLM 3**：使用WordPiece分词
  - 词汇表大小：65,024
  - 特点：针对中文优化，支持中英文混合

## 6. Tokenizer的实际应用示例

### 6.1 使用transformers库进行分词

```python
from transformers import AutoTokenizer

# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 编码文本
text = "Hello, how are you?"
encoded = tokenizer.encode(text)
print(f"编码结果: {encoded}")

# 解码回文本
decoded = tokenizer.decode(encoded)
print(f"解码结果: {decoded}")

# 详细编码信息
encoded_dict = tokenizer(text, return_tensors="pt")
print(f"详细编码: {encoded_dict}")
```

### 6.2 使用tiktoken库进行GPT分词

```python
import tiktoken

# 加载GPT-4的tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")

# 编码文本
text = "Hello, how are you?"
encoded = tokenizer.encode(text)
print(f"GPT-4编码: {encoded}")
print(f"Token数量: {len(encoded)}")

# 解码回文本
decoded = tokenizer.decode(encoded)
print(f"解码结果: {decoded}")
```

### 6.3 中文文本处理

```python
from transformers import AutoTokenizer

# 加载中文分词器
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b")

# 编码中文文本
text = "你好，世界！"
encoded = tokenizer.encode(text)
print(f"中文编码: {encoded}")
print(f"Token数量: {len(encoded)}")

# 查看每个token
for token in encoded:
    print(f"{token}: {tokenizer.decode([token])}")
```

## 7. 常见问题与解决方案

### 7.1 特殊Token设计

#### 7.1.1 常见特殊Token

| Token | 作用 | 常见模型 |
|-------|------|----------|
| `<|endoftext|>` | 文本结束标记 | GPT系列 |
| `<s>` | 序列开始标记 | LLaMA系列 |
| `</s>` | 序列结束标记 | LLaMA系列 |
| `[PAD]` | 填充标记 | BERT系列 |
| `[CLS]` | 分类标记 | BERT系列 |
| `[SEP]` | 分隔标记 | BERT系列 |
| `[MASK]` | 掩码标记 | BERT系列 |

#### 7.1.2 设计原则

- **唯一性**：每个特殊Token应有明确的唯一用途
- **不可混淆**：避免与正常词汇冲突
- **简洁性**：尽量减少特殊Token的数量

### 7.2 对齐训练/推理阶段

#### 7.2.1 常见问题

- 训练和推理时使用不同的Tokenizer配置
- 推理时遇到训练时未见过的字符
- Tokenizer的参数设置不一致

#### 7.2.2 解决方案

- 保存和加载完整的Tokenizer配置
- 确保训练和推理使用相同的词汇表
- 固定Tokenizer的参数（如`max_length`、`truncation`、`padding`）

### 7.3 中文等非空格语言处理

#### 7.3.1 挑战

- 中文没有自然的空格分隔
- 词语边界模糊，存在歧义
- 字符集庞大，词汇表设计困难

#### 7.3.2 解决方案

- 使用子词级分词算法（BPE、WordPiece等）
- 针对中文优化的词汇表设计
- 结合字符级和子词级的混合分词策略

## 8. Tokenization的局限性与挑战

### 8.1 语义碎片化

- 问题：子词分词可能破坏完整的语义单元
- 影响：导致模型难以捕捉长距离依赖关系

### 8.2 语言偏见

- 问题：词汇表设计可能引入语言偏见
- 影响：对低资源语言支持不足，产生不公平结果

### 8.3 固定粒度瓶颈

- 问题：固定的分词粒度难以适应不同的任务需求
- 影响：在细粒度和粗粒度任务间难以平衡

### 8.4 计算效率问题

- 问题：复杂的分词算法增加了模型的计算开销
- 影响：降低了模型的推理速度

## 9. 新兴方向与未来趋势

### 9.1 无Tokenizer模型

#### 9.1.1 核心思想

直接使用原始字节作为模型输入，避免显式的分词步骤。

#### 9.1.2 代表性工作

- **ByT5**：将T5模型改为直接处理字节输入
- **Mamba**：使用选择性状态空间模型处理字节序列
- **ByteLM**：基于字节的语言模型

#### 9.1.3 优势

- 消除了OOV问题
- 支持任意语言和字符集
- 简化了预处理流程

### 9.2 可学习分词

#### 9.2.1 核心思想

将分词过程作为模型的一部分进行端到端训练，学习任务相关的分词策略。

#### 9.2.2 代表性工作

- Learnable Tokenization (ACL 2023)
- Dynamic Tokenization for Efficient LLM (ICLR 2024)

#### 9.2.3 优势

- 适应不同任务的需求
- 减少语义碎片化
- 提高模型的泛化能力

### 9.3 动态分词

#### 9.3.1 核心思想

根据上下文动态调整分词粒度，在不同位置使用不同的分词策略。

#### 9.3.2 代表性工作

- Context-Aware Tokenization (EMNLP 2023)
- Adaptive Subword Regularization (ACL 2024)

#### 9.3.3 优势

- 更好地捕捉上下文信息
- 提高模型的表达能力
- 降低计算复杂度

### 9.4 多模态统一Tokenization

#### 9.4.1 核心思想

将不同模态（文本、图像、音频、视频）映射到统一的Token空间，实现多模态融合。

#### 9.4.2 代表性工作

- CLIP：文本-图像对齐
- Flamingo：多模态语言模型
- LLaVA：大语言模型多模态对齐

#### 9.4.3 优势

- 实现真正的多模态理解
- 简化模型架构
- 提高跨模态迁移能力

## 10. 总结与学习资源

### 10.1 核心要点

1. **分词是LLM的基础**：分词器位于模型架构的最前端，影响模型的整体性能
2. **子词级分词是主流**：BPE、WordPiece、Unigram等子词算法平衡了词汇表大小和语义信息
3. **不同模型选择不同的Tokenizer**：根据语言、任务和性能需求选择合适的分词器
4. **Tokenization面临挑战**：语义碎片化、语言偏见、固定粒度等问题限制了模型性能
5. **无Tokenizer模型是重要趋势**：直接处理字节输入的模型简化了预处理流程，支持任意语言

### 10.2 学习资源

#### 10.2.1 论文

- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2015) - BPE
- "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation" (Wu et al., 2016) - WordPiece
- "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing" (Kudo & Richardson, 2018)
- "ByT5: Towards a token-free future with pre-trained byte-to-byte models" (Raffel et al., 2021)

#### 10.2.2 代码库

- Hugging Face Transformers：https://github.com/huggingface/transformers
- tiktoken：https://github.com/openai/tiktoken
- SentencePiece：https://github.com/google/sentencepiece

#### 10.2.3 博客与教程

- "The Illustrated GPT-2" (Jay Alammar) - 包含BPE分词的可视化
- "How Tokenization Works in GPT Models" (OpenAI Blog)
- "Subword Tokenization for NLP" (Sebastian Ruder)

## 关联内容

- 返回[[../大语言模型核心|大语言模型核心]]主页面
- 参考[[../向量表示与Embedding/向量表示与Embedding|向量表示与Embedding]]
- 参考[[../LLM整体架构/LLM整体架构|LLM整体架构]]
- 参考[[../Transformer基础/Transformer基础|Transformer基础]]

---

*本指南基于2026年1月的最新研究编写，内容将持续更新。*