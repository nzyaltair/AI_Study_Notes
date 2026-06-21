# 向量表示与Embedding

## 目录

1. [向量表示的基础概念](#向量表示的基础概念)
2. [传统词嵌入方法](#传统词嵌入方法)
3. [上下文感知嵌入](#上下文感知嵌入)
4. [不同粒度的嵌入](#不同粒度的嵌入)
5. [LLM中的嵌入设计](#LLM中的嵌入设计)
6. [主流开源嵌入模型对比](#主流开源嵌入模型对比)
7. [嵌入质量评估与常见问题](#嵌入质量评估与常见问题)
8. [工程实践与应用](#工程实践与应用)
9. [新兴方向与未来趋势](#新兴方向与未来趋势)
10. [总结与学习资源](#总结与学习资源)

## 1. 向量表示的基础概念

### 1.1 什么是向量表示

向量表示是将文本、图像、音频等非结构化数据转换为数值向量的过程，使计算机能够理解和处理这些数据。在自然语言处理中，向量表示将词汇、句子或文档映射到高维空间中的点，从而捕获它们的语义信息。

嵌入（Embedding）是向量表示的一种具体实现，通过神经网络学习得到的低维稠密向量，能够更好地捕获数据的语义关系。

### 1.2 向量表示的发展历程

| 阶段 | 代表性方法 | 特点 |
|------|------------|------|
| 传统方法 | One-hot编码、TF-IDF | 基于统计，不包含语义信息 |
| 分布式表示 | [[Word2Vec]]、[[GloVe]]、FastText | 基于上下文，捕获语义关系 |
| 上下文感知 | ELMo、[[BERT]]、[[GPT系列]] | 基于[[预训练语言模型]]，捕获上下文信息 |
| 对比学习 | SimCSE、ConSERT | 基于[[对比学习]]，优化语义相似性 |
| 指令微调 | E5、BGE | 基于指令微调，适应下游任务 |

### 1.3 嵌入空间的性质

- **语义相似性**：语义相似的文本在嵌入空间中距离更近
- **线性关系**：某些语义关系可以通过向量运算捕获（如“国王 - 男人 + 女人 = 女王”）
- **各向异性**：嵌入向量倾向于聚集在嵌入空间的特定区域
- **稀疏性 vs 稠密性**：早期方法使用稀疏表示，现代方法普遍使用稠密表示
- **可转移性**：预训练嵌入可以迁移到不同的下游任务
- **可组合性**：不同粒度的嵌入可以组合使用

## 2. 传统词嵌入方法

### 2.1 One-hot编码

One-hot编码是最基础的向量表示方法，为每个词汇分配一个唯一的稀疏向量。

#### 2.1.1 核心思想

为每个词汇分配一个唯一的向量，其中只有一个维度为1，其余为0。

#### 2.1.2 数学形式

```
词汇表 = {"猫": 0, "狗": 1, "鱼": 2}
猫 = [1, 0, 0]
狗 = [0, 1, 0]
鱼 = [0, 0, 1]
```

#### 2.1.3 特点与局限性

- 实现简单，计算高效
- 词汇表大小庞大时，向量维度极高
- 无法捕获词汇间的语义关系
- 存在数据稀疏问题

### 2.2 Word2Vec

#### 2.2.1 核心思想

Word2Vec由Google在2013年提出，通过预测上下文或根据上下文预测目标词，学习词汇的[[分布式表示]]。

#### 2.2.2 两种训练架构

1. **CBOW（Continuous Bag of Words）**：
   - 根据上下文词汇预测目标词
   - 适合低频词，训练速度快

2. **Skip-gram**：
   - 根据目标词预测上下文词汇
   - 适合高频词，能够更好地处理生僻词

#### 2.2.3 数学形式

```
# Skip-gram 目标函数
J(θ) = -log P(w_{t-j}, ..., w_{t+j} | w_t)
```

#### 2.2.4 优化技术

- **Hierarchical Softmax**：使用哈夫曼树加速softmax计算
- **Negative Sampling**：通过负采样减少计算量

#### 2.2.5 特点与适用场景

- 能够捕获词汇间的语义关系
- 训练速度快，适合大规模语料
- 适用于大多数NLP任务

### 2.3 GloVe

#### 2.3.1 核心思想

GloVe（Global Vectors for Word Representation）由斯坦福大学在2014年提出，结合全局统计信息（词共现矩阵）和局部上下文信息，学习词汇表示。

#### 2.3.2 数学形式

```
目标函数：J = Σ_{i,j} f(X_{ij}) (w_i^T w_j + b_i + b_j - log X_{ij})^2
```

其中 `X_{ij}` 是词 `i` 和词 `j` 的共现次数，`f(X_{ij})` 是权重函数。

#### 2.3.3 特点与适用场景

- 结合了全局和局部信息
- 能够更好地捕获词汇间的线性关系
- 适合需要精确语义关系的任务

### 2.4 FastText

#### 2.4.1 核心思想

FastText由Facebook在2016年提出，将词汇分解为字符n-gram，学习字符级别的表示，从而处理未登录词。

#### 2.4.2 特点与适用场景

- 能够处理未登录词
- 适合处理形态丰富的语言（如德语、法语）
- 适合文本分类任务

## 3. 上下文感知嵌入

上下文感知嵌入是指根据词汇所处的上下文生成不同的向量表示，能够捕获词汇的多义性和上下文依赖关系。

### 3.1 ELMo

#### 3.1.1 核心思想

ELMo（Embeddings from Language Models）由Allen Institute for AI在2018年提出，基于双向LSTM的预训练语言模型，为每个词汇生成上下文相关的表示。

#### 3.1.2 架构设计

- 双向LSTM结构
- 多层表示融合
- 基于语言模型目标训练

#### 3.1.3 特点与适用场景

- 上下文相关的表示
- 能够捕获词汇的多义性
- 适合需要上下文理解的任务

### 3.2 BERT

#### 3.2.1 核心思想

BERT（Bidirectional Encoder Representations from Transformers）由Google在2018年提出，基于[[Transformer]]编码器的预训练语言模型，通过掩码语言建模和下一句预测任务学习深层上下文表示。

#### 3.2.2 嵌入设计

```
BERT输入 = Token Embedding + Segment Embedding + [[位置编码]]
```

#### 3.2.3 特点与适用场景

- 深层双向表示
- 强大的上下文理解能力
- 适合多种NLP任务，如分类、问答、命名实体识别等

### 3.3 GPT系列

#### 3.3.1 核心思想

GPT（Generative Pre-trained Transformer）系列由OpenAI提出，基于[[Transformer]]解码器的自回归预训练语言模型，生成上下文相关的表示。

#### 3.3.2 嵌入设计

- Token Embedding + [[位置编码]]
- 仅使用解码器架构
- 自回归生成

#### 3.3.3 特点与适用场景

- 强大的生成能力
- 上下文相关的表示
- 适合生成任务，如文本生成、对话生成等

### 3.4 对比学习嵌入

对比学习嵌入是通过对比学习目标优化的句子嵌入，能够更好地捕获语义相似性。

#### 3.4.1 SimCSE

##### 3.4.1.1 核心思想

SimCSE（Simple Contrastive Learning of Sentence Embeddings）由清华大学在2021年提出，通过对比学习，将相似的句子在嵌入空间中拉近，不相似的句子推开。

##### 3.4.1.2 训练方式

- 无监督：使用Dropout作为数据增强
- 有监督：使用自然语言推理（NLI）数据

##### 3.4.1.3 特点与适用场景

- 简单高效
- 无需大量标注数据
- 适合语义相似度计算任务

#### 3.4.2 ConSERT

##### 3.4.2.1 核心思想

使用多种数据增强策略，通过对比学习优化句子表示。

##### 3.4.2.2 数据增强策略

- 随机打乱词序
- 随机删除词汇
- 同义词替换
- 回译

##### 3.4.2.3 特点与适用场景

- 多种数据增强策略
- 更好的泛化能力
- 适合语义相似度计算任务

## 4. 不同粒度的嵌入

### 4.1 Token-level嵌入

#### 4.1.1 定义

为文本中的每个token生成向量表示，捕获token的语义信息和上下文信息。

#### 4.1.2 适用场景

- 命名实体识别
- 词性标注
- 关系抽取
- 机器翻译

#### 4.1.3 示例

```python
from transformers import AutoTokenizer, AutoModel

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 处理文本
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# 获取token-level嵌入
outputs = model(**inputs)
token_embeddings = outputs.last_hidden_state

print(f"Token嵌入形状: {token_embeddings.shape}")  # [1, 6, 768]
```

### 4.2 Sentence-level嵌入

#### 4.2.1 定义

为整个句子生成单一向量表示，捕获句子的整体语义。

#### 4.2.2 生成方式

- 平均池化：对token嵌入进行平均
- 最大池化：对token嵌入取最大值
- CLS token：使用BERT等模型的[CLS] token嵌入
- 专门训练的句子嵌入模型（如Sentence-BERT）

#### 4.2.3 适用场景

- 语义相似度计算
- 文本分类
- 聚类
- 信息检索

#### 4.2.4 示例

```python
from sentence_transformers import SentenceTransformer

# 加载Sentence-BERT模型
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 生成句子嵌入
sentences = ["Hello, how are you?", "I'm fine, thank you."]
embeddings = model.encode(sentences)

print(f"句子嵌入形状: {embeddings.shape}")  # [2, 384]
```

### 4.3 Document-level嵌入

#### 4.3.1 定义

为整个文档生成单一向量表示，捕获文档的主题和内容。

#### 4.3.2 生成方式

- 句子嵌入的平均池化
- 分层注意力机制
- 专门的文档嵌入模型

#### 4.3.3 适用场景

- 文档分类
- 文档聚类
- 信息检索
- 推荐系统

## 5. LLM中的嵌入设计

大型语言模型（LLM）的嵌入设计是模型性能的关键组成部分，包含多种嵌入类型的组合。

### 5.1 位置编码

#### 5.1.1 核心作用

为[[Transformer]]模型提供序列位置信息，使模型能够处理有序的序列数据。

#### 5.1.2 常见实现

- 正弦/余弦位置编码
- 可学习位置嵌入
- 旋转位置编码（[[RoPE]]）
- ALiBi（Attention with Linear Biases）

#### 5.1.3 适用场景

- 所有基于Transformer的模型
- 处理有序序列数据

### 5.2 类型嵌入（Token Type Embedding）

#### 5.2.1 核心作用

区分不同类型的token，如区分两个句子或不同的输入类型，帮助模型理解不同片段的语义角色。

#### 5.2.2 典型应用

- BERT中的句子对任务
- 多轮对话中的不同轮次
- 代码-注释对任务

### 5.3 特殊Token

特殊Token在LLM中扮演着重要角色，用于标记不同的序列结构和任务类型。

#### 5.3.1 [CLS] Token

- 用于分类任务的特殊token
- 位于序列开头
- 通常用作句子或文档的表示
- 在BERT等模型中用于下游分类任务

#### 5.3.2 [SEP] Token

- 用于分隔不同句子或段落
- 在句子对任务中使用
- 帮助模型区分不同输入片段

#### 5.3.3 [MASK] Token

- 用于[[掩码语言建模]]任务
- 随机替换部分token进行训练
- 使模型学习上下文理解能力

#### 5.3.4 [PAD] Token

- 用于填充序列到相同长度
- 不包含语义信息
- 注意力机制会忽略PAD token

## 6. 主流开源嵌入模型对比

### 6.1 模型对比

| 模型 | 维度 | 训练目标 | 主要特点 | 适用场景 |
|------|------|----------|----------|----------|
| BGE-base-en | 768 | 对比学习 + 指令微调 | 中英文支持，性能优异 | [[RAG]]、语义检索 |
| text-embedding-ada-002 | 1536 | 对比学习 | OpenAI出品，通用性强 | 各种嵌入任务 |
| GTE-large | 1024 | 对比学习 | 高性能，支持多语言 | 语义相似度、检索 |
| jina-embeddings-v2-base | 768 | 对比学习 + 指令微调 | 长上下文支持，多语言 | 长文档检索 |
| E5-large-v2 | 1024 | 指令微调 | 指令感知，任务适应能力强 | 各种嵌入任务 |
| all-MiniLM-L6-v2 | 384 | 对比学习 | 轻量级，速度快 | 对速度要求高的场景 |

### 6.2 性能对比

根据MTEB（Massive Text Embedding Benchmark）基准测试结果：

| 模型 | MTEB平均得分 | 速度（ sentences/s） |
|------|--------------|---------------------|
| BGE-large-en-v1.5 | 63.2 | 1000 |
| text-embedding-ada-002 | 62.2 | 未知 |
| GTE-large | 61.8 | 800 |
| E5-large-v2 | 61.5 | 900 |
| jina-embeddings-v2-base | 60.1 | 1200 |
| all-MiniLM-L6-v2 | 58.0 | 5000 |

## 7. 嵌入质量评估与常见问题

### 7.1 评估方法

嵌入质量评估是确保嵌入模型在下游任务中表现良好的关键步骤，需要从多个维度进行评估。

#### 7.1.1 语义文本相似度（STS）

- 评估句子对的语义相似度
- 常用数据集：STS-B、STS-DEV、STS-TEST
- 评估指标：Pearson相关系数、Spearman秩相关系数

#### 7.1.2 检索任务

- 评估检索系统的性能
- 常用指标：MRR（Mean Reciprocal Rank）、MAP（Mean Average Precision）、NDCG（Normalized Discounted Cumulative Gain）
- 常用数据集：MSMARCO、BEIR

#### 7.1.3 聚类任务

- 评估聚类结果的质量
- 常用指标：Purity、ARI（Adjusted Rand Index）、NMI（Normalized Mutual Information）
- 可视化评估：嵌入空间UMAP/t-SNE降维可视化

### 7.2 常见问题

#### 7.2.1 各向异性

**问题**：嵌入向量倾向于聚集在嵌入空间的特定区域，导致语义相似性计算不准确，模型难以区分不同语义的文本。

**解决方案**：
- 归一化嵌入向量到单位球面上
- 使用[[对比学习]]优化嵌入分布
- 应用嵌入空间正则化（如InfoNCE损失）
- 使用矩阵分解方法（如PCA）预处理嵌入

#### 7.2.2 维度坍缩

**问题**：嵌入向量的方差很小，所有向量都聚集在一起，无法区分不同的文本，导致嵌入空间失去表达能力。

**解决方案**：
- 调整模型架构和超参数（如增加隐藏层维度）
- 使用更强的正则化（如Dropout、L2正则化）
- 增加训练数据多样性
- 使用对比学习的温度参数调整

#### 7.2.3 分布偏移

**问题**：训练数据与测试数据分布不同，导致嵌入模型在新数据上表现不佳，特别是在跨领域场景中。

**解决方案**：
- 持续学习（Continual Learning）
- 领域自适应（Domain Adaptation）
- 微调预训练模型
- 使用数据增强技术

#### 7.2.4 计算复杂度

**问题**：嵌入模型的计算复杂度高，推理速度慢，难以在实时应用中部署。

**解决方案**：
- 使用轻量级模型（如all-MiniLM）
- 模型量化（INT8/INT4量化）
- 知识蒸馏（Knowledge Distillation）
- 模型剪枝（Model Pruning）
- 使用ONNX/TensorRT加速推理

## 8. 工程实践与应用

### 8.1 典型使用示例

#### 8.1.1 使用Hugging Face加载嵌入模型

```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# 生成嵌入
sentences = ["What is BERT?", "BERT is a transformer-based model."]
embeddings = model.encode(sentences)

print(f"嵌入形状: {embeddings.shape}")  # [2, 768]
```

#### 8.1.2 计算余弦相似度

```python
from sentence_transformers import util

# 计算余弦相似度
similarity = util.cos_sim(embeddings[0], embeddings[1])
print(f"余弦相似度: {similarity.item():.4f}")
```

#### 8.1.3 构建向量数据库

```python
import chromadb
from sentence_transformers import SentenceTransformer

# 初始化ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="documents")

# 加载模型
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# 文档数据
documents = [
    "BERT is a transformer-based model.",
    "GPT is a generative language model.",
    "Transformers are the backbone of modern NLP.",
    "Embeddings are vector representations of text."
]

# 生成嵌入
doc_embeddings = model.encode(documents)

# 添加到向量数据库
collection.add(
    documents=documents,
    embeddings=doc_embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# 查询
query = "What are transformers?"
query_embedding = model.encode([query])

# 检索结果
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print("检索结果:")
for doc in results["documents"][0]:
    print(f"- {doc}")
```

### 8.2 RAG场景中的嵌入选择

在[[RAG]]（检索增强生成）场景中，嵌入质量直接影响检索结果的相关性和生成质量。

#### 8.2.1 选择原则

- 语义相似度匹配能力
- 长文本处理能力
- 多语言支持
- 推理速度
- 领域适应性

#### 8.2.2 推荐模型

- 长文档（>1000 tokens）：jina-embeddings-v2-base、Longformer-Encoder-Decoder
- 高精度：BGE-large-en-v1.5、GTE-large
- 速度优先：all-MiniLM-L6-v2、paraphrase-MiniLM-L3-v2
- 多语言：BGE-m3、XLM-RoBERTa-base

#### 8.2.3 工程建议

- 对文档进行分块处理（块大小：256-1024 tokens）
- 使用混合检索策略（关键词检索 + 向量检索）
- 定期更新嵌入模型以适应新数据
- 使用嵌入缓存减少重复计算
- 评估不同嵌入模型在特定领域的表现

### 8.3 微调场景中的嵌入定制

当预训练嵌入模型在特定领域表现不佳时，需要进行微调定制以适应特定任务需求。

#### 8.3.1 微调方法

- 全参数微调：更新所有模型参数，效果最好但计算成本高
- 低秩适应（[[LoRA]]）：只训练低秩矩阵，计算成本低且效果好
- 前缀微调：只训练输入前缀，保持模型主体不变
- 适配器（Adapter）：在模型中插入小型适配器层，只训练适配器参数
- 提示微调（Prompt Tuning）：只训练提示嵌入，适合少样本场景

#### 8.3.2 数据准备

- 高质量的标注数据（通常需要1k-10k样本）
- 多样化的场景覆盖，避免过拟合
- 适当的数据增强（如回译、同义词替换）
- 正负样本均衡

#### 8.3.3 评估与部署

- 离线评估：使用标准基准测试和领域特定数据集
- 在线评估：A/B测试，比较微调前后的效果
- 模型量化：INT8/INT4量化，减少内存占用和推理延迟
- 模型压缩：结合知识蒸馏进一步减少模型大小

### 8.4 蒸馏场景中的嵌入压缩

嵌入模型蒸馏是将大模型的知识转移到小模型，在保持性能的同时降低计算成本。

#### 8.4.1 蒸馏方法

- 知识蒸馏（Knowledge Distillation）：将大模型的输出分布作为软标签训练小模型
- 对比蒸馏：保留嵌入空间的语义关系和对比结构
- 量化蒸馏：结合量化和蒸馏，同时优化模型大小和精度
- 注意力蒸馏：迁移大模型的注意力权重信息

#### 8.4.2 实现示例

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# 加载教师模型和学生模型
teacher_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
student_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 准备训练数据
train_examples = [
    InputExample(texts=["This is a sentence.", "This is a similar sentence."]),
    InputExample(texts=["This is a sentence.", "This is a very different sentence."])
]

# 数据加载器
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

# 定义损失函数 - 使用对比损失进行蒸馏
loss_function = losses.MSELoss(model=student_model)

# 蒸馏训练
student_model.fit(
    train_objectives=[(train_dataloader, loss_function)],
    epochs=3,
    warmup_steps=100,
    teacher_model=teacher_model,  # 指定教师模型
    output_path="./distilled-model"
)
```

## 9. 新兴方向与未来趋势

### 9.1 动态嵌入

#### 9.1.1 核心思想

动态嵌入（query-aware embeddings）根据查询内容动态调整文档嵌入，生成更适合特定查询的表示，解决传统静态嵌入的局限性。

#### 9.1.2 代表性工作

- query-aware embeddings（EMNLP 2023）
- dynamic pooling（ACL 2024）
- adaptive embeddings with attention gates（NeurIPS 2023）

#### 9.1.3 应用场景

- 复杂查询的信息检索
- 个性化推荐系统
- 多轮对话系统
- 上下文感知的RAG系统

### 9.2 多模态对齐嵌入

#### 9.2.1 核心思想

将不同模态（文本、图像、音频、视频）的嵌入对齐到同一共享空间，实现跨模态理解和生成。

#### 9.2.2 代表性工作

- CLIP（OpenAI，ICML 2021）：文本-图像对齐
- DALL-E 3（OpenAI，2023）：文本-图像生成
- Flamingo（DeepMind，ICML 2022）：多模态理解
- LLaVA（UCSD，NeurIPS 2023）：大语言模型多模态对齐

#### 9.2.3 应用场景

- 跨模态检索（如以文搜图、以图搜文）
- 多模态生成（如文本生成图像、图像生成文本）
- 视觉问答（VQA）
- 多模态RAG系统

### 9.3 可编辑/可控嵌入

#### 9.3.1 核心思想

允许用户编辑或控制嵌入空间中的特定属性（如情感、主题、风格），实现可控的文本生成和理解。

#### 9.3.2 代表性工作

- Diffusion-based Embedding Editing（ICLR 2024）
- Attribute-Guided Embedding Manipulation（ACL 2023）
- Counterfactual Embedding Generation（EMNLP 2023）

#### 9.3.3 应用场景

- 可控内容生成
- 个性化推荐
- 安全生成（避免有害内容）
- 内容风格转换

### 9.4 嵌入在新兴架构中的角色

#### 9.4.1 MoE架构

- 动态路由：根据输入嵌入选择不同的专家网络
- 专家特定嵌入：为不同专家网络学习特定的嵌入表示
- 混合专家嵌入：结合多个专家的嵌入输出

#### 9.4.2 长上下文建模

- 分层嵌入：对不同层级的上下文学习不同的嵌入表示
- 稀疏注意力嵌入：只关注相关上下文的嵌入
- 记忆增强嵌入：结合外部记忆系统的嵌入

#### 9.4.3 Agent记忆系统

- 长期记忆嵌入：将长期知识编码为嵌入向量
- 检索增强记忆：通过向量检索从记忆中获取相关信息
- 动态更新记忆：根据新信息不断更新记忆嵌入

### 9.5 近2-3年顶会重要进展

- **ACL 2024**："Efficient Long-Context Embeddings with Sparse Attention" - 提出了一种高效的长上下文嵌入方法
- **EMNLP 2023**："Dynamic Embedding Adaptation for Cross-Domain Retrieval" - 提出了动态嵌入适应方法
- **NeurIPS 2023**："MoE-Embeddings: Sparse Expert Embeddings for Large Language Models" - 提出了MoE架构下的嵌入设计
- **ICLR 2024**："Aligning Embeddings Across Languages and Modalities with Contrastive Learning" - 提出了跨语言跨模态嵌入对齐方法

### 9.6 未来展望

- 嵌入层将变得更加轻量化和高效，适应边缘计算场景
- 多模态对齐嵌入将成为主流，实现真正的跨模态理解
- 动态嵌入将显著提升RAG和对话系统的性能
- 嵌入将在Agent系统中扮演核心角色，连接感知、记忆和决策
- 可解释性嵌入将得到更多关注，提高模型的透明度

## 10. 总结与学习资源

### 10.1 核心要点

1. **向量表示的发展**：从传统的One-hot编码到现代的上下文感知嵌入，向量表示技术不断演进，捕获越来越丰富的语义信息。

2. **上下文感知是关键**：上下文感知嵌入（如BERT、GPT系列）能够捕获词汇的多义性和上下文信息，在各种NLP任务中表现优异。

3. **对比学习的兴起**：对比学习（如SimCSE）通过将相似的句子在嵌入空间中拉近，不相似的句子推开，显著提升了语义相似度计算的性能。

4. **嵌入质量评估**：评估嵌入质量需要考虑多个方面，包括语义文本相似度、检索性能和聚类性能等。

5. **工程实践建议**：在实际应用中，需要根据具体场景选择合适的嵌入模型，并考虑嵌入质量、计算效率和部署成本等因素。

6. **未来发展趋势**：动态嵌入、多模态对齐嵌入、可编辑/可控嵌入以及嵌入在新兴架构中的应用是未来的重要发展方向。

### 10.2 学习资源

#### 10.2.1 论文

- "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013) - Word2Vec
- "GloVe: Global Vectors for Word Representation" (Pennington et al., 2014) - GloVe
- "Attention Is All You Need" (Vaswani et al., 2017) - Transformer
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018) - BERT
- "SimCSE: Simple Contrastive Learning of Sentence Embeddings" (Gao et al., 2021) - SimCSE
- "E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training" (Wang et al., 2023) - E5
- "BGE: Towards Better General Embeddings" (Zhang et al., 2023) - BGE

#### 10.2.2 代码库

- SentenceTransformers：https://github.com/UKPLab/sentence-transformers
- Hugging Face Transformers：https://github.com/huggingface/transformers
- ChromaDB：https://github.com/chroma-core/chroma
- FAISS：https://github.com/facebookresearch/faiss

#### 10.2.3 博客与教程

- "The Illustrated Transformer" (Jay Alammar)
- "A Visual Guide to Using BERT for the First Time" (Jay Alammar)
- "Sentence Embeddings: A Survey" (Lilian Weng)
- "Contrastive Learning for NLP" (Sebastian Ruder)

## 关联内容

- 返回[[../大语言模型核心|大语言模型核心]]主页面
- 参考[[../位置编码/位置编码|位置编码]]
- 参考[[../Transformer基础/Transformer基础|Transformer]]
- 参考[[../RAG/RAG|RAG]]
- 参考[[../对比学习/对比学习|对比学习]]

---

*本指南基于2026年1月的最新研究编写，内容将持续更新。*