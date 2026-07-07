---
title: "从零构建 GPT - Transformer 架构详解"
tags:
  - 深度学习
  - 语言模型
  - NLP
  - PyTorch
  - Transformer
  - 自注意力
  - GPT
  - 神经网络
  - 注意力机制
aliases:
  - Let's build GPT
  - Karpathy GPT
  - Transformer 从零构建
  - nanoGPT 教程
  - 自注意力机制
created: 2026-07-07
---

# 从零构建 GPT - Transformer 架构详解

## 核心概述

本笔记整理自 Andrej Karpathy 的 "Let's build GPT: from scratch, in code, spelled out" 课程。本课程从零开始，逐步构建一个基于 Transformer 架构的字符级语言模型，最终训练出一个能生成莎士比亚风格文本的 GPT 模型。

**为什么重要**：Transformer 架构源自 2017 年的论文 *"Attention Is All You Need"*，最初为机器翻译设计，却在随后五年间席卷整个 AI 领域。GPT（Generative Pre-trained Transformer）的核心正是这套架构。理解 Transformer 的内部机制是理解 ChatGPT 等现代大语言模型的基础。

**解决什么问题**：
- 理解 ChatGPT 背后的核心神经网络架构
- 掌握自注意力机制的工作原理
- 理解 Transformer 各组件（多头注意力、前馈网络、残差连接、层归一化、Dropout）如何协作
- 从零实现一个可工作的 GPT 模型，而非黑盒调用

> [!note] 核心论点
> Transformer 的本质是**通信（自注意力）与计算（前馈网络）的交替堆叠**。通过逐步添加位置编码、因果掩码、多头注意力、残差连接、层归一化和 Dropout，我们从最简单的二元语言模型（验证损失 2.45）逐步构建到完整的 Transformer（验证损失 1.48）。整个模型约 1000 万参数，在莎士比亚数据集上训练约 15 分钟即可生成具有莎士比亚风格的文本。

---

## 知识体系

### 1. 背景与动机

#### 1.1 从 ChatGPT 到 Transformer

ChatGPT 是一个**概率系统**——对同一提示词可生成多种不同回应。它本质上是一个**语言模型**，预测词语（更广泛地说是字符或 token）之间的排列方式。

GPT = **Generative Pre-trained Transformer**（生成式预训练 Transformer）

- **Generative**：自回归地生成文本
- **Pre-trained**：在大规模文本上预训练
- **Transformer**：核心神经网络架构

Transformer 源自 2017 年 Google 的论文 *"Attention Is All You Need"*。该论文原为机器翻译设计，但 Transformer 架构随后被广泛应用于各种 AI 场景。

> [!info]nanoGPT 项目
> Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 是最简洁的 GPT 实现之一，仅由两个约 300 行代码的文件组成：`model.py`（模型定义）和 `train.py`（训练逻辑）。

#### 1.2 数据集：tiny shakespeare

本课程使用 tiny shakespeare 数据集——约 1MB 的莎士比亚作品全集。目标：建模字符间的序列关系，让 Transformer 学习生成莎士比亚风格文本。

---

### 2. 数据准备与分词

#### 2.1 字符级分词器

最简单的分词方案：将每个字符映射为一个整数。

```python
# 读取数据
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 构建词表
chars = sorted(list(set(text)))
vocab_size = len(chars)  # 65 个字符

# 编码器与解码器
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]        # 字符串 → 整数列表
decode = lambda l: ''.join([itos[i] for i in l]) # 整数列表 → 字符串
```

> [!tip] 分词器的选择
> 字符级分词器词表小（65 个字符），但序列很长。实际应用中常用子词分词器：
> - **SentencePiece**（Google）：基于子词
> - **tiktoken**（OpenAI）：字节对编码（BPE），GPT-2 使用约 5 万个 token
> 
> 词表大小与序列长度之间存在权衡：词表越大，序列越紧凑。

#### 2.2 训练集与验证集

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 前 90% 训练，后 10% 验证
train_data = data[:n]
val_data = data[n:]
```

验证集的作用：判断模型是否过拟合（死记硬背 vs 真正学会规律）。

#### 2.3 块大小与批次维度

Transformer 不会一次性处理整个文本，而是处理固定长度的**块（block）**。

**关键设计**：一个长度为 `block_size + 1` 的文本片段实际包含 `block_size` 个训练样本：

```
上下文: [18]            → 目标: 47
上下文: [18, 47]        → 目标: 56
上下文: [18, 47, 56]    → 目标: 57
...以此类推
```

> [!important] 为何在 1 到 block_size 的所有长度上训练
> 让 Transformer 适应从 1 到 `block_size` 的任意长度上下文。推理时只需一个字符即可启动生成，逐步扩展到完整块。超过块大小时需截断。

**批次维度**：为充分利用 GPU 并行能力，将多个独立序列打包为一个批次。各序列之间完全独立，互不影响。

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
```

---

### 3. 二元语言模型（Baseline）

#### 3.1 最简实现

从最简单的模型开始：仅根据单个 token 预测下一个 token。

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 每个token直接查表得到logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) → logits: (B, T, vocab_size)
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # PyTorch 交叉熵要求 (N, C) 格式
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
```

> [!warning] PyTorch 交叉熵的维度要求
> `F.cross_entropy` 期望输入形状为 `(N, C)`，其中 C 是通道（类别）维度。如果输入是 `(B, T, C)`，需 reshape 为 `(B*T, C)`，目标 reshape 为 `(B*T,)`。

#### 3.2 初始损失分析

随机初始化时，预期损失约为 $-\ln(1/65) \approx 4.17$。实际初始损失 4.87，说明初始化并非完全均匀分布。

#### 3.3 文本生成

```python
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        # 只关注最后一个时间步的预测
        logits = logits[:, -1, :]            # (B, C)
        probs = F.softmax(logits, dim=-1)     # (B, C)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx
```

#### 3.4 训练脚本的关键改进

将 Notebook 代码转换为脚本 `bigram.py`（约 120 行），加入以下改进：

1. **GPU 支持**：`device = 'cuda' if torch.cuda.is_available() else 'cpu'`
2. **损失估计函数**：多次采样取平均，减少批次噪声
3. **`model.eval()` / `model.train()`**：养成管理模型模式的习惯（当前模型虽无差异，但未来会用到）
4. **`@torch.no_grad()`**：评估时不计算梯度，节省内存

二元模型训练后验证损失约 2.45。**核心局限**：token 之间没有信息交互，仅凭单个字符预测下一个字符。

---

### 4. 自注意力的数学基础

在实现自注意力之前，需要掌握一个核心数学技巧：**用矩阵乘法实现加权聚合**。

#### 4.1 问题的本质

序列中第 5 个 token 只能与第 1~4 个 token 交流（信息只能从过去流向现在）。最简单的交互方式：对前面所有 token 取**平均值**。

#### 4.2 从循环到矩阵乘法

**低效版本**（循环）：

```python
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]        # (t, C)
        xbow[b, t] = xprev.mean(0) # 平均
```

**向量化版本**（矩阵乘法）：

核心洞察：**下三角矩阵乘法 = 对历史元素求和**。

```python
# 下三角矩阵（归一化后每行和为1）
wei = torch.zeros((T, T))
wei = wei.masked_fill(torch.tril(T, T) == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
# wei 现在是一个归一化的下三角矩阵

# 批量矩阵乘法实现加权聚合
xbow2 = wei @ x  # (T,T) @ (B,T,C) → (B,T,C) via broadcast
```

> [!important] 批量矩阵乘法（bmm）
> 当 `wei` 是 `(T, T)` 而 `x` 是 `(B, T, C)` 时，PyTorch 会自动在 `wei` 前面添加批次维度，使其变为 `(B, T, T)`，然后对每个批次元素独立执行 `(T,T) @ (T,C) → (T,C)`。

#### 4.3 三种等价实现

1. **循环求平均**
2. **矩阵乘法**（下三角全 1 矩阵归一化）
3. **Softmax 掩码**（零初始化 + 下三角掩码 + softmax）

第三种方式最关键——它将权重初始化为零，通过 softmax 归一化得到均匀分布。**但这些权重不必是固定的**——它们可以由数据动态决定。这正是自注意力的核心。

---

### 5. 自注意力机制

#### 5.1 Query、Key、Value

自注意力让 token 间的关注度由**数据驱动**而非固定。

每个 token 产生三个向量：

| 向量 | 含义 | 直觉 |
|------|------|------|
| **Query (Q)** | 我想找什么 | "我是元音，想找前面的辅音" |
| **Key (K)** | 我有什么 | "我是辅音，在前四位" |
| **Value (V)** | 我要传递的信息 | 被聚合的实际内容 |

**注意力得分** = Query 与 Key 的点积。点积越大，说明两者越"匹配"，关注度越高。

```python
class Head(nn.Module):
    """单个自注意力头"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # 注意力得分
        wei = q @ k.transpose(-2, -1)  # (B, T, T)

        # 因果掩码：未来不能影响过去
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # 缩放 + Softmax
        wei = wei * (head_size ** -0.5)  # 缩放注意力
        wei = F.softmax(wei, dim=-1)

        # 加权聚合
        out = wei @ v  # (B, T, head_size)
        return out
```

> [!note] 为什么叫"自"注意力
> 因为 Query、Key、Value 都来自同一个输入 X。X 同时产生键、查询和值，节点进行自注意力运算。
> 
> 对比**交叉注意力**（Cross-Attention）：Query 来自一处，Key 和 Value 来自另一处（如编码器输出），用于从独立的信息源中提取信息。

#### 5.2 缩放点积注意力（Scaled Dot-Product Attention）

> [!important] 为何要除以 $\sqrt{d_k}$
> 如果输入 Q、K 服从标准高斯分布（均值 0，方差 1），点积 $Q \cdot K$ 的方差为 $d_k$（头维度大小）。
> 
> **不缩放的问题**：当 $d_k$ 较大时，点积值方差大，导致 softmax 输出趋向 one-hot（过于集中于最大值）。这意味着每个节点只从另一个节点收集信息，梯度消失，无法有效学习。
> 
> **缩放后**：$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$，方差恢复为 1，softmax 输出更均匀，尤其在初始化阶段至关重要。

#### 5.3 注意力机制的关键性质

1. **信息传递机制**：注意力本质上有向图中节点间的信息传递。每个节点通过指向它的所有节点加权汇总来聚合信息，权重由数据动态决定。

2. **无位置感知**：注意力默认不知道节点的位置——它只是处理一组向量。因此需要**位置编码**注入位置信息。这与卷积不同，卷积核天然依赖空间位置。

3. **批次内独立**：批次内各样本的 token 只在自身序列内部交流，不会跨样本交互。

4. **有向图视角**：语言建模使用自回归有向图（第 $i$ 个节点被 $1, 2, \ldots, i$ 指向）。但注意力原则上可用于任意有向图结构。

---

### 6. 位置编码与 Token 嵌入

#### 6.1 双重嵌入

```python
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)   # Token 身份
self.position_embedding_table = nn.Embedding(block_size, n_embd) # 位置信息

def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)     # (B, T, n_embd)
    pos_emb = self.position_embedding_table(torch.arange(T))  # (T, n_embd)
    x = tok_emb + pos_emb  # 广播相加：token身份 + 位置信息
```

引入 `n_embd`（嵌入维度）作为中间层，不再直接从嵌入表输出 logits，而是通过一个线性层（`lm_head`）将嵌入映射到词表维度。

> [!warning] 生成时的块大小限制
> 由于使用了位置嵌入表（大小为 `block_size`），生成时输入长度不能超过 `block_size`。需在生成函数中截断：`idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]`

---

### 7. 多头注意力

#### 7.1 核心思想

使用多个并行的小型注意力头，每个头关注不同模式（如元音-辅音关系、位置模式等），最后沿通道维度拼接。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
```

#### 7.2 维度设计

当嵌入维度 `n_embd = 32`，使用 4 个头时，每个头大小为 8（$32 / 4 = 8$）。4 个头各输出 8 维，拼接后恢复为 32 维。

> [!tip] 类比分组卷积
> 多头注意力类似于分组卷积——不用一个大卷积核处理所有通道，而是分组并行处理，每组关注不同特征子空间。

---

### 8. 前馈网络

自注意力负责 token 间的**通信**，前馈网络负责每个 token 独立的**计算**。

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
```

> [!note] 通信与计算的交替
> Transformer 的核心设计理念：**通信（自注意力）与计算（前馈网络）交替进行**。
> - 自注意力：token 间交换信息（"互相看了一眼"）
> - 前馈网络：每个 token 独立处理收到的信息（"细品对方的意思"）

---

### 9. Transformer Block

将多头注意力和前馈网络组合为一个可重复的**块**：

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # 预归一化
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # 残差 + 预归一化
        x = x + self.ffwd(self.ln2(x))
        return x
```

---

### 10. 残差连接

#### 10.1 原理

残差连接（Residual Connection / Skip Connection）源自 2015 年的 ResNet 论文。核心思想：信号可以通过加法直接从输入传到输出，绕过中间变换。

```
输入 x ──────────────────── (+) ──── 输出
         ↘                 ↗
          变换(注意力/FFN)
```

```python
x = x + self.sa(self.ln1(x))  # 残差路径 + 分支计算
```

#### 10.2 为什么有效

- **梯度高速公路**：反向传播时，加法运算将梯度均匀分配给两个分支。存在一条从损失到输入的无阻碍梯度通道。
- **渐进式上线**：残差块初始化时对路径贡献很小，梯度直接畅通流动。随着训练进行，残差块逐渐"上线"开始发挥作用。
- **解决深度网络优化难题**：使深层网络可优化。

> [!note] 残差块的初始化
> 在 nanoGPT 的 `model.py` 中，残差投影层使用特殊的缩放初始化：`std = 0.02 / sqrt(2 * n_layer)`，确保初始时残差块贡献很小。

---

### 11. 层归一化

#### 11.1 BatchNorm vs LayerNorm

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | 列（跨样本） | 行（单个样本内部） |
| 依赖批次 | 是（需维护运行均值/方差） | 否 |
| 训练/推理差异 | 有 | 无 |
| 缓冲区 | 需要运行均值/方差 | 不需要 |
| 可训练参数 | $\gamma$, $\beta$ | $\gamma$, $\beta$ |

**LayerNorm 对单个样本的特征维度进行归一化**，不跨样本计算。训练和推理行为一致，无需维护运行时缓冲区。

#### 11.2 预归一化 vs 后归一化

原始 Transformer 论文使用**后归一化**（Post-Norm）：

```python
x = LayerNorm(x + SubLayer(x))
```

现代实践通常采用**预归一化**（Pre-Norm），效果更好：

```python
x = x + SubLayer(LayerNorm(x))
```

> [!important] 预归一化的优势
> 预归一化使梯度在残差路径上更直接地流动，有利于深层网络的优化。过去 5 年间，这成为 Transformer 实践中的一项重要改进。

---

### 12. Dropout

Dropout 源自 2014 年论文，是一种正则化技术：

- **训练时**：随机关闭一部分神经元（设为 0），每次前向/反向传播屏蔽的部分不同
- **推理时**：所有神经元激活
- **效果**：相当于训练多个子网络的集成，测试时整合为一个整体

在本模型中，Dropout 应用于：
1. 自注意力权重（softmax 后）
2. 残差路径返回前（投影后）

```python
self.attn_dropout = nn.Dropout(dropout)
self.resid_dropout = nn.Dropout(dropout)
```

---

### 13. 完整 Transformer 模型

#### 13.1 模型结构总览

```
输入 idx (B, T)
    │
    ├── Token Embedding (vocab_size → n_embd)
    ├── Position Embedding (block_size → n_embd)
    │         ↓ 相加
    │     x (B, T, n_embd)
    │         ↓ Dropout
    │
    ├── Block 1: [LN → MultiHeadAttn → Residual] → [LN → FFN → Residual]
    ├── Block 2: [LN → MultiHeadAttn → Residual] → [LN → FFN → Residual]
    ├── ...
    ├── Block N: [LN → MultiHeadAttn → Residual] → [LN → FFN → Residual]
    │
    ├── Final LayerNorm
    └── Linear (n_embd → vocab_size) → logits
```

#### 13.2 完整实现

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)           # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T))  # (T, n_embd)
        x = tok_emb + pos_emb                          # (B, T, n_embd)
        x = self.blocks(x)                             # 通过所有 Block
        x = self.ln_f(x)                               # 最终归一化
        logits = self.lm_head(x)                       # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

#### 13.3 前馈网络的 4 倍扩展

参考原始论文：前馈层内部维度为嵌入维度的 **4 倍**（如 512 → 2048）。

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 扩展 4 倍
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # 投影回 n_embd
            nn.Dropout(dropout),
        )
```

---

### 14. 损失演进与模型扩展

#### 14.1 渐进式改进过程

| 阶段 | 模型改进 | 验证损失 |
|------|----------|----------|
| 二元模型 | 仅 Token Embedding | ~2.45 |
| + 自注意力（单头） | Token + Position Embedding + 1 个 Head | ~2.40 |
| + 多头注意力 | 4 个并行 Head | ~2.28 |
| + 前馈网络 | 通信 + 计算交替 | ~2.24 |
| + 残差连接 + 投影 | 深层可优化 | ~2.08 |
| + 层归一化 | 预归一化 | ~2.06 |
| + Dropout + 扩展规模 | 384 维, 6 层, 6 头 | **1.48** |

#### 14.2 最终模型超参数

```python
batch_size = 64
block_size = 256        # 上下文长度从 8 扩展到 256
n_embd = 384            # 嵌入维度
n_head = 6              # 注意力头数（每个头 64 维）
n_layer = 6             # Transformer Block 层数
dropout = 0.2           # 20% dropout
learning_rate = 3e-4
```

> [!info] 训练规模
> - 约 1000 万参数
> - 约 100 万字符（100 万 token）的训练数据
> - RTX 2000 GPU 上约 15 分钟
> - 验证损失 1.48（从 2.07 显著改善）

---

### 15. Transformer 架构全景：编码器、解码器与交叉注意力

#### 15.1 三种 Transformer 变体

原始论文 *"Attention Is All You Need"* 采用**编码器-解码器**结构，用于机器翻译。但实际应用中有三种变体：

| 变体 | 特征 | 应用场景 |
|------|------|----------|
| **Encoder-only** | 无掩码，所有 token 自由交流 | 情感分析、文本分类、BERT |
| **Decoder-only** | 因果掩码，自回归生成 | 语言建模、GPT 系列 |
| **Encoder-Decoder** | 编码器 + 交叉注意力 + 解码器 | 机器翻译、T5 |

#### 15.2 编码器 vs 解码器

```
编码器 (Encoder)              解码器 (Decoder)
┌─────────────┐              ┌─────────────────┐
│ 自注意力     │              │ 掩码自注意力     │
│ (无掩码)     │              │ (三角掩码)       │
│ 所有token    │              │ 只看过去        │
│ 自由交流     │              │                 │
└─────────────┘              └─────────────────┘
```

本课程实现的是**纯解码器**Transformer：
- 使用三角掩码防止未来 token 影响过去 token
- 自回归生成：可从模型直接采样
- 与 GPT 架构一致

> [!important] 为何使用纯解码器
> 我们只是在生成文本，没有任何外部条件约束，目标仅是模仿给定数据集。纯解码器通过三角掩码实现自回归特性，非常适合语言建模。

#### 15.3 交叉注意力

在编码器-解码器架构中，解码器不仅有掩码自注意力，还有**交叉注意力**：

- **自注意力**：Q、K、V 都来自同一输入 X
- **交叉注意力**：Q 来自解码器，K 和 V 来自编码器输出

交叉注意力用于从一组独立节点（如编码器编码的源语言句子）中提取信息，整合到当前节点（解码器生成目标语言）。

---

### 16. nanoGPT 代码解析

#### 16.1 与课程代码的对比

nanoGPT 的 `model.py` 与课程中实现的模型几乎相同，但有一些工程优化：

| 特性 | 课程代码 | nanoGPT |
|------|----------|---------|
| 多头注意力 | 显式创建多个 Head 对象 | 批量实现，4 维张量 |
| 激活函数 | ReLU | GELU（为加载 OpenAI 检查点） |
| 注意力实现 | 手动矩阵乘法 | Flash Attention（PyTorch ≥ 2.0） |
| 权重绑定 | 无 | Token Embedding 与 LM Head 共享权重 |
| 初始化 | 简单 | 残差投影层缩放初始化 |
| 优化器 | 简单 AdamW | 分组权重衰减 + Fused AdamW |

#### 16.2 CausalSelfAttention 的高效实现

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 所有头的 Q, K, V 一次投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()
        # 一次投影得到 Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # reshape 为 (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 重新拼接所有头
        y = self.resid_dropout(self.c_proj(y))  # 输出投影
        return y
```

#### 16.3 权重绑定

```python
self.transformer.wte.weight = self.lm_head.weight
```

Token Embedding 矩阵和最终输出投影矩阵共享权重——因为将 token 映射到向量和将向量映射回 token 概率分布是对偶操作。

#### 16.4 Block 结构（预归一化）

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 预归一化 + 残差
        x = x + self.mlp(self.ln_2(x))    # 预归一化 + 残差
        return x
```

---

### 17. ChatGPT 的训练流程

ChatGPT 的训练分为两大阶段：**预训练**和**微调（对齐）**。

#### 17.1 预训练阶段

在大规模互联网文本上训练纯解码器 Transformer，目标是生成连贯文本——本质上是文档补全工具。

| 参数 | 课程模型 | GPT-3 |
|------|----------|-------|
| 参数量 | ~10M | 175B |
| Token 数 | ~30 万 | 3000 亿 |
| 架构 | 几乎相同 | 几乎相同 |
| 训练硬件 | 单 GPU | 数千 GPU |

> [!warning] 预训练后不是助手
> 预训练后的模型只会补全文档。当你提问时，它可能续写更多问题或忽略你的提问——因为训练目标是"续写序列"，而非"回答问题"。

#### 17.2 微调阶段（对齐）

让模型从"文档补全工具"转变为"问答助手"。OpenAI 的方法包含三步：

1. **监督微调（SFT）**
   - 收集问答格式的训练数据（数千例）
   - 微调模型使其在问题下方生成答案
   - 大模型在少量数据上微调效果显著

2. **奖励模型训练（RM）**
   - 让模型生成多个候选回复
   - 评分者按偏好排序候选回复
   - 训练奖励模型预测回复的受青睐程度

3. **PPO 强化学习**
   - 使用奖励模型作为奖励信号
   - 通过 PPO（近端策略优化）算法优化模型生成策略
   - 使模型回复获得更高奖励值

> [!note] nanoGPT 的定位
> nanoGPT 专注于**预训练阶段**——这正是本课程所覆盖的内容。微调和对齐阶段的代码和数据大多未公开。

---

### 18. 关键概念总结

#### 18.1 注意力机制的本质

注意力是一种**通用的信息传递机制**，可用于任意有向图结构：
- **自注意力**：Q、K、V 来自同一来源
- **交叉注意力**：Q 来自一处，K、V 来自另一处
- **编码器注意力**：无掩码，所有节点自由交流
- **解码器注意力**：因果掩码，只看过去

#### 18.2 Transformer 的设计哲学

```
通信 ←→ 计算 ←→ 通信 ←→ 计算 ←→ ...
(注意力)  (前馈)  (注意力)  (前馈)
```

每个 Block 做两件事：
1. **通信**：通过多头自注意力让 token 交换信息
2. **计算**：通过前馈网络让每个 token 独立处理信息

#### 18.3 深度网络可优化的三大法宝

1. **残差连接**：梯度高速公路，初始化时残差块几乎不存在
2. **层归一化**：预归一化让梯度更直接流动
3. **Dropout**：正则化，防止过拟合

#### 18.4 缩放点积注意力的完整公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $QK^T$：计算查询与键的匹配度（注意力得分矩阵）
- $\sqrt{d_k}$：缩放因子，控制方差
- $\text{softmax}$：归一化为概率分布
- $V$：按注意力权重加权聚合值

---

## 参考资源

- **论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- **论文**：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al., 2015) — 残差连接
- **论文**：[Layer Normalization](https://arxiv.org/abs/1607.06450) (Ba et al., 2016)
- **论文**：[Dropout](https://jmlr.org/papers/v15/srivastava14a.html) (Srivastava et al., 2014)
- **代码**：[nanoGPT](https://github.com/karpathy/nanoGPT)
- **相关笔记**：[[makemore - 字符级语言模型与二元语法模型]]、[[makemore - 激活函数与梯度及批量归一化]]、[[makemore - 构建WaveNet分层网络]]
