# LLM核心架构

## 背景

大语言模型的架构以Decoder-only Transformer为核心，经过GPT系列、LLaMA系列、Qwen系列的不断演进，形成了现代LLM的标准架构设计。理解LLM架构是理解模型能力边界和优化方向的基础。

## 核心思想

通过Scaling Laws指导下的架构设计，以自回归方式预测下一个token，使模型在足够规模下涌现出通用语言理解和生成能力。

## 技术原理

### 标准LLM架构

```
Input Tokens → Embedding + RoPE → [N × Transformer Block] → RMSNorm → Output Projection
                                    ├─ Multi-Head Attention (Causal)
                                    ├─ Residual + RMSNorm
                                    ├─ SwiGLU FFN
                                    └─ Residual + RMSNorm
```

### Scaling Laws

- **OpenAI Scaling Law**：损失 $L(N, D) \propto N^{-0.076} + D^{-0.095}$（N=参数量, D=数据量）
- **Chinchilla Scaling**（DeepMind）：最优计算下 $D/N \approx 20$，即每个参数应训练约20个token
- **涌现能力**：某些能力（如CoT推理、上下文学习）在模型规模超过阈值后突然出现

### 位置编码：RoPE

旋转位置编码（Rotary Position Embedding）是现代LLM的标配：

- 通过旋转矩阵将位置信息编码到Q和K中
- $\mathbf{q}_m^T \mathbf{k}_n = g(\mathbf{q}, \mathbf{k}, m-n)$，只依赖相对位置
- 支持长度外推（配合NTK-aware scaling等技巧）
- LLaMA/Qwen/DeepSeek等主流模型均采用

### 混合专家模型（MoE）

将FFN替换为多个专家网络，通过路由器选择Top-K专家：

$$\text{MoE}(x) = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

- **稀疏激活**：总参数量大，但每次推理只用部分参数
- **优势**：以较低计算成本实现更大模型容量
- **挑战**：负载均衡、通信开销、训练不稳定性
- **代表模型**：Mixtral、DeepSeek-MoE、Qwen-MoE

### 注意力机制变体

| 变体 | 原理 | 适用场景 |
|------|------|---------|
| **标准注意力** | $O(n^2)$ | 短序列 |
| **Flash Attention** | 分块计算，减少内存读写 | 通用加速 |
| **滑动窗口注意力** | 局部注意力 | 长序列 |
| **GQA/MQA** | 共享K/V头组 | 推理加速 |
| **线性注意力** | 核函数近似 | 超长序列 |

### 长上下文机制

| 技术 | 原理 | 代表 |
|------|------|------|
| **位置插值** | 缩放RoPE频率 | LLaMA-2-32K |
| **NTK-aware** | 调整RoPE基频 | YaRN |
| **Ring Attention** | 分布式长序列 | Claude 100K+ |
| **KV-Cache压缩** | 淘汰不重要的KV | StreamingLLM |

### 主流开源模型架构

| 模型系列 | 架构特点 | 位置编码 | FFN | Norm |
|---------|---------|---------|-----|------|
| **LLaMA** | 标准Decoder | RoPE | SwiGLU | RMSNorm |
| **Qwen** | 类LLaMA + GQA | RoPE | SwiGLU | RMSNorm |
| **DeepSeek** | MoE变体 | RoPE | SwiGLU | RMSNorm |
| **Mistral** | 滑动窗口+GQA | RoPE | SwiGLU | RMSNorm |

## 发展演进

GPT-1（2018，117M）→ GPT-2（2019，1.5B）→ GPT-3（2020，175B）→ Chinchilla（2022，70B）→ LLaMA（2023）→ GPT-4（2023）→ MoE普及（2024）→ 推理模型o1（2024）

## 与其他技术关系

- LLM架构基于[[Transformer架构详解|Transformer]]，采用Decoder-only路线
- 位置编码和注意力机制是核心创新点
- 训练流程详见[[LLM训练与对齐]]
- 推理时的KV-Cache和量化详见[[LLM推理与优化]]
- MoE的稀疏思想与[[集成学习方法|集成学习]]有相似之处
- 长上下文机制与[[Agent智能体|RAG]]系统互补
