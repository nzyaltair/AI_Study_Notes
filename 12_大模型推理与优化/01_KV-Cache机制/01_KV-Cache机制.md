# KV-Cache 机制

## 背景与发展

在 [[注意力机制]] 的自回归解码中，生成第 $n$ 个 token 需要基于前 $n-1$ 个 token 的信息。如果不做缓存，每生成一个新 token 都要重新计算所有历史 token 的 Key 和 Value，导致大量重复计算，计算复杂度为 $O(n^2)$。

KV-Cache（Key-Value Cache）通过缓存已计算的注意力键值对，在后续解码步骤中复用，将每步计算量从 $O(n^2)$ 降为 $O(n)$。这是 [[大模型推理]] 中最基础也最关键的优化技术。

随着模型上下文长度从 2K 扩展到 128K+，KV-Cache 的显存占用成为新的瓶颈，催生了 PagedAttention、KV-Cache 量化、StreamingLLM 等一系列优化技术。

## 核心思想

在 Transformer 解码过程中，KV-Cache 位于每层的多头注意力模块。生成第 $t$ 个 token 时：

1. 只计算当前 token 的 Query（Q）、Key（K）、Value（V）
2. 将当前 token 的 K 和 V 追加到缓存中
3. 用当前 Q 与缓存中所有 K 计算注意力，再与缓存中所有 V 加权求和

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 Q 是当前 token 的查询（维度 $1 \times d_k$），K 和 V 是缓存中所有历史 token 的键值对（维度 $t \times d_k$ 和 $t \times d_v$）。

## 技术原理

### 显存占用公式

KV-Cache 的显存占用为：

$$M_{\text{KV}} = 2 \times n_{\text{layers}} \times \text{batch} \times n_{\text{heads}} \times \text{seq\_len} \times d_{\text{head}} \times \text{dtype\_size}$$

其中因子 2 表示同时缓存 K 和 V。

以 Llama-2-70B（FP16）为例：$n_{\text{layers}}=80$，$n_{\text{heads}}=64$，$d_{\text{head}}=128$，$\text{dtype\_size}=2$ 字节。当 seq_len=2048、batch=1 时：

$$M = 2 \times 80 \times 1 \times 64 \times 2048 \times 128 \times 2 \approx 5.36 \text{ GB}$$

### GQA/MQA 对缓存的影响

- **MHA**（Multi-Head Attention）：每个注意力头有独立的 K/V，缓存最大
- **MQA**（Multi-Query Attention）：所有头共享一组 K/V，缓存减少 $n_{\text{heads}}$ 倍
- **GQA**（Grouped-Query Attention）：将头分组，组内共享 K/V，在 MHA 和 MQA 之间权衡

### 张量布局

KV-Cache 的典型张量布局为每层存储 K 和 V 两个张量：

- K: `[batch_size, num_heads, seq_len, head_dim]`
- V: `[batch_size, num_heads, seq_len, head_dim]`

共 $2 \times n_{\text{layers}}$ 个张量。预分配策略在序列开始时分配最大长度的缓存，内存利用率低但无分配开销；动态扩展策略按需分配，利用率高但有性能开销。

### PagedAttention

借鉴操作系统虚拟内存管理思想，将 KV-Cache 划分为固定大小的块（Pages），通过页表管理不连续的内存块：

- 每个序列的 KV-Cache 由一组逻辑块组成，映射到物理块
- 消除内存碎片，利用率提升 60-70%
- 支持变长序列和连续批处理（新请求可动态加入）
- 减少 OOM 风险
- 支持前缀共享：相同系统 prompt 的请求共享 KV-Cache 前缀

### KV-Cache 量化

将 KV-Cache 从 FP16 量化为 INT8 或 INT4：
- FP16 -> INT8：显存减少 50%
- FP16 -> INT4：显存减少 75%
- 需要处理注意力分数的精度损失，通常对 K 和 V 使用不同的量化策略

### 缓存驱逐策略

长上下文场景下，需驱逐部分 KV 以控制显存：

- **StreamingLLM**：保留初始 token（Attention Sink）+ 最近 token 的 KV，实现恒定内存下的无限长度推理。Attention Sink 机制发现初始 token 在注意力中承担"汇聚"作用，移除会导致注意力分数坍塌
- **H2O**（Heavy-Hitter Oracle）：基于注意力分数动态淘汰不重要的 KV，保留对生成贡献最大的 token
- **滑动窗口**：只保留最近 W 个 token 的 KV，简单高效但丢失早期上下文

## 发展演进

MHA（标准注意力）-> KV-Cache 缓存 -> MQA/GQA（减少 KV 头数）-> PagedAttention（分页管理）-> KV-Cache 量化 -> StreamingLLM（缓存驱逐）

## 关键算法·模型

- **PagedAttention**（vLLM, Kwon et al. 2023）：分页 KV-Cache 管理，吞吐量提升 2-4x
- **StreamingLLM**（Xiao et al. 2023）：Attention Sink 机制，支持无限长度推理
- **MQA**（Shazeer 2019）：多查询注意力，所有头共享 KV
- **GQA**（Ainslie et al. 2023）：分组查询注意力，Llama-2 采用

## 应用场景

- **自回归推理**：所有 LLM 生成任务的基础优化，几乎被所有推理框架默认采用
- **高并发服务**：PagedAttention 支持 vLLM 的高吞吐服务，单机可服务数千并发请求
- **长上下文推理**：缓存驱逐策略支持 100K+ token 生成，StreamingLLM 支持无限长度流式输出
- **边缘部署**：KV-Cache 量化减少设备显存压力，使 7B 模型可在 8GB 显卡上运行
- **多模态推理**：多模态模型同样使用 KV-Cache 缓存视觉和文本 token 的键值对

## 与其他技术关系

- KV-Cache 是 [[注意力机制]] 在推理阶段的工程实现
- [[量化]] 技术可应用于 KV-Cache 以减少显存
- PagedAttention 是 [[vLLM]] 的核心创新
- 缓存驱逐与 [[长上下文扩展]] 密切相关
- GQA/MQA 影响模型架构设计，参见 [[大语言模型]]
