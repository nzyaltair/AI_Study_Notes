# Transformer架构详解

## 背景

2017年Google论文《Attention Is All You Need》提出了Transformer架构，彻底改变了深度学习。它摒弃了[[序列模型|RNN]]的递归计算，完全基于注意力机制，实现了序列的并行处理。Transformer是[[大语言模型|LLM]]的核心架构，也是现代深度学习的基石。

## 核心思想

**自注意力机制**：序列中每个位置直接与所有位置交互，捕捉全局依赖关系。通过多头注意力和位置编码的组合，Transformer实现了强大的序列建模能力。

## 技术原理

### 自注意力机制

给定查询(Q)、键(K)、值(V)，计算注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q = XW_Q$，$K = XW_K$，$V = XW_V$（Self-Attention中QKV来自同一输入）
- $\sqrt{d_k}$ 缩放因子：防止点积过大导致softmax梯度消失
- 复杂度：$O(n^2 \cdot d)$，n为序列长度

### 多头注意力

将QKV投影到h个不同的子空间，分别计算注意力后拼接：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

- 每个头关注不同的子空间信息（如语法关系、语义关系）
- 典型设置：d_model=512, h=8, d_k=64

### 位置编码

Transformer本身是排列不变的，需要位置编码注入顺序信息。

| 类型 | 方法 | 代表模型 |
|------|------|---------|
| **正弦编码** | $\sin/\cos$固定编码 | 原始Transformer |
| **可学习编码** | 位置嵌入表可训练 | GPT、ViT |
| **相对位置编码** | 编码token间相对距离 | T5、Transformer-XL |
| **RoPE** | 旋转位置编码 | LLaMA、现代LLM |
| **ALiBi** | 注意力偏置 | BLOOM |

### 编码器-解码器架构

**编码器**（BERT使用）：
- 多头自注意力（无掩码）→ Add & LayerNorm → FFN → Add & LayerNorm
- N层堆叠，双向注意力

**解码器**（GPT使用）：
- 掩码多头自注意力（因果掩码）→ Add & LayerNorm → FFN → Add & LayerNorm
- N层堆叠，单向注意力（只看左侧）

**编码器-解码器**（T5使用）：
- 解码器额外有交叉注意力层（Cross-Attention）

### 前馈神经网络（FFN）

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

现代变体：
- **GELU**：$\text{FFN}(x) = \text{GELU}(xW_1)W_2$
- **SwiGLU**（LLaMA）：$\text{FFN}(x) = (\text{Swish}(xW_1) \otimes xW_3)W_2$，需3个权重矩阵

### 残差连接与层归一化

- **残差连接**：$y = x + \text{SubLayer}(x)$，解决深层网络梯度问题
- **层归一化**：对每个样本的特征维度归一化
- **归一化位置**：
  - Post-LN（原始）：LayerNorm在残差之后
  - Pre-LN（现代）：LayerNorm在残差之前，训练更稳定
  - RMSNorm：LayerNorm简化版，现代LLM标配

## 发展演进

注意力机制（2014）→ Seq2Seq+Attention → **Transformer**（2017）→ BERT/GPT（2018）→ T5（2019）→ GPT-3（2020）→ [[大语言模型|LLM]]时代

Transformer从机器翻译工具发展为AI的基础架构，扩展到视觉（ViT）、语音、[[多模态模型|多模态]]等领域。

## 关键变体

| 变体 | 创新 | 用途 |
|------|------|------|
| **BERT** | 双向编码器 | 理解类任务 |
| **GPT** | 单向解码器 | 生成类任务 |
| **T5** | 编码器-解码器 | 统一框架 |
| **ViT** | 图像分块→Transformer | 计算机视觉 |
| **Efficient Transformers** | 稀疏注意力 | 长序列 |

## 应用领域

- **NLP**：机器翻译、文本生成、问答、摘要
- **计算机视觉**：ViT、DETR
- **语音**：Whisper、Conformer
- **[[多模态模型|多模态]]**：CLIP、Flamingo
- **科学**：AlphaFold 2

## 与其他技术关系

- Transformer解决了[[序列模型|RNN]]的并行化和长期依赖问题
- 自注意力源于Seq2Seq的注意力机制
- [[训练技巧与稳定性|训练技巧]]（LayerNorm、Warmup）对Transformer尤为重要
- 从零实现详见[[从零构建GPT]]（karpathy教程）
- Transformer是[[预训练语言模型|预训练语言模型]]的架构基础
- Transformer的变体扩展到[[多模态模型|多模态]]领域
