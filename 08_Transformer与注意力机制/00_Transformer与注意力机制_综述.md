# 方向八：Transformer与注意力机制

## 定义与范围

基于自注意力机制的深度网络架构，实现并行化序列建模。Transformer已成为自然语言处理、计算机视觉和多模态AI的主流架构。

## 历史演进

| 年代 | 里程碑 | 意义 |
|------|--------|------|
| 2014 | Bahdanau Attention | 注意力机制首次提出 |
| 2015 | Luong Attention | 注意力机制改进 |
| 2017 | Transformer（Vaswani等） | 自注意力架构革命 |
| 2018 | BERT / GPT | 预训练Transformer |
| 2020 | FlashAttention | 注意力计算IO优化 |
| 2021 | Linear Attention | 线性注意力探索 |
| 2022 | FlashAttention-2 | 进一步优化 |
| 2023 | Grouped-Query Attention | 推理效率优化 |
| 2024 | FlashAttention-3 / MLA | Hopper架构优化/DeepSeek注意力 |
| 2025 | 混合架构（Jamba等） | 注意力与SSM融合 |

## 核心概念与方法

- **注意力机制**：Query-Key-Value、缩放点积注意力、注意力权重
- **多头注意力**：并行多组注意力、子空间表示
- **位置编码**：正弦编码、学习式编码、相对位置编码、RoPE、ALiBi
- **Transformer架构**：Encoder-Decoder、Encoder-only、Decoder-only
- **效率优化**：FlashAttention、稀疏注意力、线性注意力、分组查询注意力
- **交叉注意力**：Q来自一个序列，K/V来自另一个——多模态和Encoder-Decoder的核心
- **注意力共享**：MQA（多查询）、GQA（分组查询）

## 核心公式推导

### 1. Self-Attention缩放点积

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

缩放因子 $\sqrt{d_k}$ 推导：$QK^T$ 元素方差为 $d_k$，除以 $\sqrt{d_k}$ 归一化到1。

### 2. Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

总参数量 $= 4d^2$，与单头注意力相同。

### 3. RoPE（旋转位置编码）

$$q_m' = R_m q_m, \quad k_n' = R_n k_n$$

$$q_m'^T k_n' = q_m^T R_m^T R_n k_n = q_m^T R_{n-m} k_n$$

注意力仅依赖相对位置 $m-n$。

### 4. FlashAttention分块计算

空间复杂度从 $\mathcal{O}(n^2)$ 降至 $\mathcal{O}(n)$，通过tiling和online softmax实现2-4倍加速。

## 代表论文与里程碑

- Vaswani et al. (2017): Attention Is All You Need
- Devlin et al. (2018): BERT
- Radford et al. (2018): GPT
- Su et al. (2021): RoFormer（RoPE）
- Dao et al. (2022): FlashAttention
- Ainslie et al. (2023): GQA

## 学习资源

- The Annotated Transformer（Harvard NLP）
- Andrej Karpathy: Let's build GPT from scratch
- Jay Alammar: The Illustrated Transformer
- FlashAttention论文及开源代码

## 笔记要点与实践任务

- 从零实现Self-Attention和Multi-Head Attention
- 实现并对比不同位置编码（正弦/RoPE/ALiBi）的外推性能
- 推导并实现FlashAttention的分块算法
- 实现Grouped-Query Attention并测量推理加速
- 分析注意力矩阵的秩与信息瓶颈

## 与其他方向关系

- 「数学基础」中的softmax和矩阵运算是其计算基础
- Transformer取代了「序列模型」中的RNN/LSTM
- 是「预训练语言模型」和「大语言模型」的核心架构
- 注意力机制被广泛应用于「计算机视觉」（ViT）和「多模态AI」
