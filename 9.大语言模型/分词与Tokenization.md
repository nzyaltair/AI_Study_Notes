# 分词与Tokenization

## 背景

分词（Tokenization）是将文本转换为模型可处理的基本单元（token）的过程，是LLM的第一步。分词质量直接影响模型性能、词表大小和多语言能力。BPE是现代LLM的主流分词算法。

## 核心思想

在字符级和词级分词之间找到平衡：既能表示任意文本（包括未登录词），又能控制词表大小和序列长度。

## 技术原理

### 分词算法对比

| 算法 | 原理 | 代表模型 | 特点 |
|------|------|---------|------|
| **BPE** | 从字符开始，迭代合并最高频pair | GPT系列 | 主流 |
| **WordPiece** | 类似BPE，基于似然选择pair | BERT | Google系 |
| **SentencePiece** | 直接处理原始文本，支持BPE/Unigram | LLaMA/T5 | 多语言友好 |
| **Unigram** | 从大词表逐步删除，最大化似然 | T5/ALBERT | 概率模型 |

### BPE算法流程

1. 初始化：将所有词拆分为字符序列
2. 统计所有相邻字符pair的频率
3. 合并频率最高的pair，加入词表
4. 重复2-3直到达到目标词表大小

```python
# BPE 核心逻辑伪代码
vocab = list(all_characters)
while len(vocab) < target_vocab_size:
    pairs = count_pairs(corpus)  # 统计相邻token对频率
    best_pair = max(pairs, key=pairs.get)
    new_token = best_pair[0] + best_pair[1]
    vocab.append(new_token)
    corpus = merge_pair(corpus, best_pair, new_token)
```

### 词表设计考量

- **词表大小**：通常32K-128K（GPT-2: 50K, LLaMA: 32K, Qwen: 152K）
- **多语言**：多语言模型需要更大词表以覆盖不同字符集
- **特殊token**：`<bos>`, `<eos>`, `<pad>`, `<unk>`及Chat模板token
- **数字处理**：单独数字token（如T5）vs 拆分数字

### Tokenization的挑战

- **编码不一致**：同一文本不同tokenizer结果不同
- **前导空格**：`" world"` vs `"world"`可能映射为不同token
- **大小写**：区分大小写的tokenizer可能浪费词表
- **代码处理**：缩进、特殊符号的tokenization
- **多语言混合**：中英混合文本的token效率

### Token效率

不同语言的token效率差异显著：
- 英文：~4字符/token
- 中文：~1-2字符/token（取决于词表设计）
- 代码：变化大，取决于缩进和符号

这直接影响[[LLM推理与优化|推理成本]]和上下文窗口利用率。

## 发展演进

字符级 → 词级 → BPE（2016）→ WordPiece → SentencePiece → BBPE（Byte-level BPE, GPT-2）→ 多语言优化

## 应用领域

- **LLM预处理**：所有LLM的输入处理第一步
- **词表设计**：影响模型多语言能力和推理效率
- **上下文窗口**：分词效率直接影响可用上下文长度
- **成本控制**：token数量直接决定API计费

## 与其他技术关系

- 分词是[[LLM核心架构|LLM架构]]的输入处理环节
- Token效率影响[[LLM推理与优化|推理成本]]和上下文利用
- 从零构建BPE分词器详见[[分词器 - 从零构建BPE分词器与Tokenization详解]]
- 分词是[[预训练语言模型|预训练]]和[[LLM训练与对齐|微调]]的前置步骤
- 向量表示与Embedding建立在分词之上
