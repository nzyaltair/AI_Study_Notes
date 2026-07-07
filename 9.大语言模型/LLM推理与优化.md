# LLM推理与优化

## 背景

LLM推理面临三大挑战：显存占用大（模型参数+KV-Cache）、计算延迟高（自回归生成）、成本高昂（GPU资源）。推理优化技术是LLM从实验室走向生产的关键，直接影响用户体验和运营成本。

## 核心思想

通过KV-Cache减少重复计算、通过量化减少显存占用、通过并行化和调度策略提升吞吐量、通过解码策略控制生成质量。

## 技术原理

### KV-Cache机制

自回归生成时，缓存已计算的Key和Value，避免重复计算：

- 生成第t个token时，只需计算新token的Q与缓存中所有K的注意力
- **显存**：KV-Cache大小 = $2 \times n_{layers} \times n_{heads} \times d_{head} \times \text{seq\_len} \times \text{batch} \times \text{dtype\_size}$
- **优化**：GQA/MQA减少KV头数、PagedAttention分页管理

### 量化技术

| 方法 | 精度 | 特点 | 工具 |
|------|------|------|------|
| **GPTQ** | 4-bit | 逐层量化，基于二阶信息 | AutoGPTQ |
| **AWQ** | 4-bit | 激活感知量化，保护重要权重 | AutoAWQ |
| **GGUF** | 2-8-bit | CPU/GPU混合， llama.cpp生态 | llama.cpp |
| **INT8** | 8-bit | 简单高效，精度损失小 | bitsandbytes |
| **FP8** | 8-bit浮点 | 硬件原生支持，H100 | TensorRT-LLM |

**量化原理**：将FP16权重映射到低精度：$w_{quant} = \text{round}(w / s) \times s$，其中 $s$ 为缩放因子

### 推理加速框架

| 框架 | 核心技术 | 适用场景 |
|------|---------|---------|
| **vLLM** | PagedAttention、连续批处理 | 通用LLM服务 |
| **TensorRT-LLM** | 内核融合、INT8/FP8 | NVIDIA最优性能 |
| **llama.cpp** | CPU/GPU混合、GGUF量化 | 本地部署、边缘设备 |
| **SGLang** | RadixAttention、结构化生成 | Agent应用 |
| **DeepSpeed-FastGen** | Dynamic Splitfuse | 深度学习生态 |

### 连续批处理（Continuous Batching）

传统批处理需等同一batch所有请求完成，连续批处理在每步动态加入新请求、移除已完成请求，大幅提升GPU利用率。

### 解码与采样策略

| 策略 | 原理 | 特点 |
|------|------|------|
| **Greedy** | 每步选概率最高token | 确定性，易重复 |
| **Temperature** | $P' = \text{softmax}(\logits / T)$ | T<1更确定，T>1更多样 |
| **Top-K** | 只从概率最高的K个token采样 | 控制多样性 |
| **Top-P**（核采样） | 从累积概率≥P的最小集合采样 | 自适应多样性 |
| **Beam Search** | 维护多个候选序列 | 适合翻译，不适合对话 |
| **Speculative Decoding** | 小模型草稿+大模型验证 | 加速2-3倍 |

### 长上下文优化

- **KV-Cache压缩**：StreamingLLM（保留首尾token的KV）
- **滑动窗口注意力**：只关注最近W个token
- **量化KV-Cache**：将KV-Cache量化到INT8/FP8

### 成本与延迟权衡

| 优化方向 | 技术 | 效果 |
|---------|------|------|
| **降低显存** | 量化、GQA、KV压缩 | 2-4x显存节省 |
| **提升吞吐** | 连续批处理、Tensor Parallel | 3-10x吞吐提升 |
| **降低延迟** | Speculative Decoding、Flash Attention | 2-3x延迟降低 |
| **降低成本** | 模型蒸馏、路由小模型 | 5-10x成本降低 |

## 发展演进

KV-Cache → Flash Attention → 量化（GPTQ/AWQ/GGUF）→ PagedAttention/vLLM → Speculative Decoding → FP8推理

## 应用领域

- **LLM服务**：API服务（如OpenAI、vLLM部署）
- **本地部署**：llama.cpp在消费级GPU/CPU上运行
- **边缘部署**：手机端LLM（GGUF量化）
- **高并发场景**：连续批处理提升服务吞吐

## 与其他技术关系

- 推理优化依赖[[LLM核心架构|LLM架构]]的理解（注意力机制、KV-Cache）
- 量化与[[训练技巧与稳定性|混合精度训练]]的技术原理相通
- 模型部署详见[[容器化与Docker]]和系统工程化模块
- [[ONNX模型导出|ONNX]]是跨框架推理的中间格式
- 解码策略影响[[Agent智能体|Agent]]的输出质量
