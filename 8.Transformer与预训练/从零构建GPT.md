# 从零构建GPT

## 背景

本笔记整理自 Andrej Karpathy 的系列教程，从零构建理解GPT的核心机制。通过从最简单的二元语言模型逐步构建到完整的Transformer架构，深入理解[[Transformer架构详解|Transformer]]的内部工作原理。

> **核心论点**：Transformer的本质是**通信（自注意力）与计算（前馈网络）的交替堆叠**。通过逐步添加各组件，我们从验证损失2.45的二元模型构建到验证损失1.48的完整GPT。

## Karpathy 系列教程索引

| 教程 | 核心内容 | 对应笔记 |
|------|---------|---------|
| **Micrograd** | 从零构建自动微分引擎 | [[Micrograd - 从零构建自动微分引擎与神经网络]] |
| **makemore Part 1-5** | 字符级语言模型→MLP→激活函数→反向传播→WaveNet | [[makemore - 字符级语言模型与二元语法模型]] |
| **从零构建GPT** | Transformer架构详解、nanoGPT实现 | [[从零构建GPT - Transformer架构详解]] |
| **GPT的现状** | 训练流程与应用实践 | [[GPT的现状 - 训练流程与应用实践]] |
| **分词器** | BPE分词器与Tokenization | [[分词器 - 从零构建BPE分词器与Tokenization详解]] |
| **复现GPT2** | 从零训练1.24亿参数模型 | [[复现GPT2 - 从零训练1.24亿参数模型]] |

## 核心知识链

### 1. 自动微分基础（Micrograd）

- 从零实现反向传播引擎
- 理解计算图、梯度链式法则
- 构建`Value`类支持自动求导
- 与[[深度学习框架|PyTorch]]的autograd原理一致

### 2. 语言模型演进（makemore）

从最简单的语言模型逐步进化：

```
二元语法模型（Bagram）
  → 多层感知机（MLP + 字符嵌入）
  → 激活函数与梯度（初始化、BatchNorm）
  → 手动反向传播（成为反向传播忍者）
  → WaveNet（分层网络）
```

### 3. Transformer核心（从零构建GPT）

逐步构建GPT的关键组件：

1. **自注意力机制**：$\text{softmax}(QK^T/\sqrt{d_k})V$
2. **位置编码**：注入序列顺序信息
3. **因果掩码**：确保只关注左侧token
4. **多头注意力**：多子空间并行注意力
5. **残差连接 + LayerNorm**：稳定深层训练
6. **前馈网络（FFN）**：逐位置非线性变换

### 4. 训练实践

- **数据**：tiny shakespeare（~1MB，65个字符）
- **模型**：~1000万参数
- **训练**：~15分钟（GPU）
- **结果**：验证损失1.48，生成莎士比亚风格文本

### 5. 分词器（Tokenization）

- BPE（Byte Pair Encoding）分词原理
- 从字符级到子词级的过渡
- 与[[分词与Tokenization|分词与Tokenization]]的关系

### 6. 复现GPT-2

- 1.24亿参数模型从零训练
- OpenWebText数据集
- 扩展到完整GPT-2规模

## nanoGPT 项目

Karpathy的[nanoGPT](https://github.com/karpathy/nanoGPT)是最简洁的GPT实现：

- `model.py`（~300行）：模型定义
- `train.py`（~300行）：训练逻辑
- 支持从字符级到GPT-2规模的训练

## 与其他技术关系

- 理解[[Transformer架构详解|Transformer架构]]的最佳实践路径
- 从零实现依赖[[深度学习基础|神经网络基础]]和[[优化方法|微积分]]
- 训练技巧与[[训练技巧与稳定性|训练稳定性]]一致
- nanoGPT是理解[[大语言模型|LLM]]内部机制的教学工具
- 分词器详见[[分词与Tokenization]]
