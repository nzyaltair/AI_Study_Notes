# 分词与Tokenization

## 背景与发展

分词（Tokenization）是将文本转换为模型可处理的基本单元（token）的过程，是LLM的第一步。分词质量直接影响模型性能、词表大小和多语言能力。BPE是现代LLM的主流分词算法。

## 核心思想

在字符级和词级分词之间找到平衡：既能表示任意文本（包括未登录词），又能控制词表大小和序列长度。子词分词（subword tokenization）是这一平衡的最佳解。

## 技术原理

### 分词算法对比

| 算法 | 原理 | 代表模型 | 特点 |
|------|------|---------|------|
| BPE | 从字符开始，迭代合并最高频pair | GPT系列 | 主流 |
| WordPiece | 类似BPE，基于似然选择pair | BERT | Google系 |
| SentencePiece | 直接处理原始文本，支持BPE/Unigram | LLaMA/T5 | 多语言友好 |
| Unigram | 从大词表逐步删除，最大化似然 | T5/ALBERT | 概率模型 |

### BPE算法流程

1. 初始化：将所有词拆分为字符序列
2. 统计所有相邻字符pair的频率
3. 合并频率最高的pair，加入词表
4. 重复2-3直到达到目标词表大小

```python
vocab = list(all_characters)
while len(vocab) < target_vocab_size:
    pairs = count_pairs(corpus)
    best = max(pairs, key=pairs.get)
    corpus = merge_pair(corpus, best)
    vocab.append(best[0] + best[1])
```

### Byte-level BPE (BBPE)

GPT-2引入，以字节而非字符为基本单元，可表示任意UTF-8文本，彻底消除未登录词问题。

### 词表设计考量

- **词表大小**：通常32K-128K（GPT-2: 50K, LLaMA: 32K, Qwen: 152K）
- **多语言**：多语言模型需要更大词表以覆盖不同字符集
- **特殊token**：`<bos>`, `<eos>`, `<pad>`, `<unk>`及Chat模板token
- **数字处理**：单独数字token（T5）vs 拆分数字

### Token效率

不同语言的token效率差异显著，直接影响推理成本和上下文窗口利用率：

- 英文：~4字符/token
- 中文：~1-2字符/token（取决于词表设计）
- 代码：变化大，取决于缩进和符号

## 发展演进

字符级 → 词级 → BPE（2016）→ WordPiece → SentencePiece → BBPE（GPT-2, 2019）→ 多语言优化

## 关键算法·模型

- **GPT系列tokenizer**：BBPE，词表50K+
- **LLaMA tokenizer**：SentencePiece BPE，词表32K
- **Qwen tokenizer**：多语言优化BBPE，词表152K
- **tiktoken**：OpenAI的高效BPE实现

## 应用场景

- **LLM预处理**：所有LLM的输入处理第一步
- **词表设计**：影响模型多语言能力和推理效率
- **上下文窗口**：分词效率直接影响可用上下文长度
- **成本控制**：token数量直接决定API计费

## 与其他技术关系

- 分词是[[LLM整体架构]]的输入处理环节
- Token效率影响推理成本和上下文利用
- 是[[预训练语言模型]]和微调的前置步骤
- [[Embedding]]建立在分词之上
