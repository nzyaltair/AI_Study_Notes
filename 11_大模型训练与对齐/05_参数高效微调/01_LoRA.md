# LoRA

## 背景与发展

LoRA（Low-Rank Adaptation）由 Hu et al. (2021, Microsoft) 提出，是当前最广泛使用的参数高效微调方法。LoRA 基于预训练权重更新具有低秩特性的假设，通过低秩分解将可训练参数降至约 1%，同时保持接近全参数微调的性能。后续发展出 AdaLoRA、DoRA、PiSSA 等改进变体。

## 核心思想

预训练模型的权重更新矩阵 $\Delta W$ 具有低秩特征——微调过程中权重的变化集中在低维子空间。因此可以用两个低秩矩阵的乘积 $BA$ 来近似 $\Delta W$，仅训练 $A$ 和 $B$ 而保持原始权重 $W$ 不变。

## 技术原理

### 数学定义

对于原始权重矩阵 $W \in \mathbb{R}^{d \times d}$，LoRA 引入：

- $A \in \mathbb{R}^{r \times d}$：随机初始化（高斯分布）
- $B \in \mathbb{R}^{d \times r}$：初始化为零矩阵

因此初始时 $BA = 0$，模型行为与原始模型一致。微调时仅更新 $A$ 和 $B$：

$$W' = W + \Delta W = W + BA$$

其中秩 $r \ll d$（如 $r=16$, $d=4096$），可训练参数从 $d^2$ 降至 $2dr$。

### 前向传播

$$y = Wx + BAx = (W + BA)x$$

### 反向传播

$$\frac{\partial L}{\partial A} = B^T \frac{\partial L}{\partial y} x^T, \quad \frac{\partial L}{\partial B} = \frac{\partial L}{\partial y} x^T A^T$$

$W$ 的梯度被忽略（冻结），仅计算 $A$、$B$ 的梯度。

### 缩放因子

实际使用中引入缩放因子 $\alpha$：

$$\Delta W = \frac{\alpha}{r} BA$$

$\alpha$ 通常设为 $2r$，使得 $\Delta W = 2BA$，控制 LoRA 的贡献度。

### 插入位置

- **注意力层 Q、V 投影**：最常用，参数量最小
- **Q + K + V + O 全投影**：性能更好，参数量增加
- **FFN 层线性变换**：进一步增强表达能力
- **所有线性层**：性能最佳，参数量最大

### 与全参数微调的关系

当 $r = d$ 时，$BA$ 可以表示任意 $d \times d$ 矩阵，LoRA 等价于全参数微调。$r$ 越大表达能力越强但参数量越多，需要在性能和效率间权衡。

### 推理时合并

LoRA 的关键优势：推理时可将 $BA$ 合并到 $W$ 中得到 $W' = W + BA$，不增加任何推理延迟。

## 发展演进

LoRA (2021, Microsoft) → LoRA+ (2023) → AdaLoRA (2023，动态秩) → DoRA (2024，动态缩放) → PiSSA (2024，SVD 初始化)

## 关键算法·模型

- **LoRA**：低秩适配，PEFT 的基础方法
- **AdaLoRA**：训练过程中自动裁剪不重要的秩
- **DoRA**：每层动态缩放，减少超参数调整
- **PiSSA**：用 SVD 分解初始化 LoRA 矩阵，加速收敛

## 应用场景

- **大模型微调**：7B-70B 模型的高效微调
- **多任务共享**：多个 LoRA 适配器共享同一基座模型
- **快速实验**：低资源下快速尝试不同任务
- **持续学习**：通过添加新 LoRA 学习新任务

## 与其他技术关系

- LoRA 是 [[参数高效微调]] 的核心方法
- [[QLoRA]] 是 LoRA + 4-bit 量化的扩展
- [[Adapter]] 是另一种 PEFT 方案，结构不同但目标一致
- LoRA 可与 [[监督微调]]、[[DPO]] 等训练阶段结合
