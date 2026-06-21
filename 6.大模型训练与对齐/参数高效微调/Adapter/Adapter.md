# Adapter（适配器）

## 1. 简介

Adapter（适配器）是一种[[参数高效微调|参数高效微调（PEFT）]]方法，通过在预训练模型的特定位置插入小型可训练模块（适配器），仅调整这些适配器的参数来实现模型微调。这种方法在保持预训练模型参数固定的同时，能够高效地将模型适配到新任务。

%% 核心思想：在不修改预训练模型主体的情况下，通过插入小型可训练模块来实现任务适配 %% ^core-idea

## 2. 核心原理与架构

### 2.1 原始设计（Houlsby et al. 2019）

Houlsby 等人在 2019 年提出的原始 Adapter 设计具有以下特点：

- **瓶颈结构**：适配器通常采用 "压缩-扩展" 结构，即先通过一个降维线性层，再经过非线性激活，最后通过一个升维线性层恢复原始维度
- **残差连接**：适配器模块通常包含残差连接，允许梯度直接流过预训练模型
- **激活函数**：常用 GELU 或 ReLU 作为激活函数
- **层归一化**：在适配器前后添加层归一化，帮助稳定训练

### 2.2 典型架构

```
原始 Transformer 层：
输入 → 注意力层 → Dropout → 残差连接 → 层归一化 → FFN层 → Dropout → 残差连接 → 层归一化 → 输出

插入 Adapter 的层：
输入 → 注意力层 → Dropout → 残差连接 → 层归一化 → Adapter1 → FFN层 → Dropout → 残差连接 → 层归一化 → Adapter2 → 输出
```

### 2.3 插入位置

Adapter 可以插入到 Transformer 模型的不同位置：

- **FFN 层后**：最常见的位置，对模型性能影响较小
- **注意力层后**：可以更直接地影响模型的注意力机制
- **多头注意力内部**：在每个注意力头后插入适配器
- **模型并行路径**：在模型的并行计算路径中插入适配器

## 3. Adapter 变体

### 3.1 Parallel Adapter

- **核心思想**：将适配器与主模型并行连接，而非串行插入
- **优势**：减少推理延迟，因为适配器计算可以与主模型并行进行
- **结构**：
  ```
  输入 → 主模型路径 →
                 → 聚合 → 输出
       → 适配器路径 →
  ```

### 3.2 Compacter

- **核心思想**：通过低秩分解和共享参数进一步减少适配器参数量
- **特点**：
  - 使用低秩矩阵分解减少线性层参数量
  - 跨层共享适配器参数
  - 引入深度可分离卷积减少计算量

### 3.3 AdapterFusion

- **核心思想**：学习如何融合多个预训练适配器的输出，实现多任务知识迁移
- **应用场景**：当有多个相关任务的适配器可用时，可以通过 AdapterFusion 动态组合它们
- **实现**：引入门控机制，学习每个适配器的权重

### 3.4 MAD-X Adapter

- **核心思想**：为不同语言或模态设计特定的适配器，实现跨语言/跨模态迁移
- **特点**：
  - 语言特定适配器 + 任务特定适配器的双层结构
  - 支持零样本跨语言迁移

## 4. 与其他 PEFT 方法对比

| 方法 | 参数量 | 训练效率 | 任务迁移性 | 推理延迟 | 实现复杂度 |
|------|--------|----------|------------|----------|------------|
| [[Adapter]] | 少 | 中 | 高 | 中 | 中 |
| [[LoRA]] | 极少 | 高 | 中 | 低 | 低 |
| [[Prefix Tuning]] | 极少 | 低 | 中 | 低 | 中 |
| [[IA³]] | 极少 | 高 | 中 | 低 | 低 |
| [[Prompt Tuning]] | 极少 | 低 | 低 | 低 | 低 |

## 5. 工程实现

### 5.1 PyTorch 实现示例

```python
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual

# 在 Transformer 层中插入 Adapter
class TransformerLayerWithAdapter(nn.Module):
    def __init__(self, original_layer, adapter_dim=64):
        super().__init__()
        self.original_layer = original_layer
        self.adapter = Adapter(original_layer.output_dim, adapter_dim)
    
    def forward(self, x, attention_mask=None):
        x = self.original_layer(x, attention_mask=attention_mask)
        x = self.adapter(x)
        return x
```

### 5.2 Hugging Face PEFT 库使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel, get_peft_model, AdapterConfig

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 配置 Adapter
adapter_config = AdapterConfig(
    adapter_type="houlsby",
    adapter_size=64,
    dropout=0.1,
    non_linearity="gelu",
    task_type="CAUSAL_LM"
)

# 应用 Adapter
model = get_peft_model(model, adapter_config)

# 查看可训练参数
model.print_trainable_parameters()
```

### 5.3 DeepSpeed 集成

- **优势**：利用 DeepSpeed 的 ZeRO 优化器进一步减少内存使用
- **配置**：
  ```json
  {
    "zero_optimization": {
      "stage": 2,
      "offload_optimizer": {
        "device": "cpu"
      }
    }
  }
  ```

## 6. 适用场景与常见陷阱

### 6.1 适用场景

- **多任务学习**：为每个任务训练一个适配器，共享同一基础模型
- **低资源微调**：在资源有限的情况下微调大型模型
- **模型服务中动态切换适配器**：根据不同用户需求动态加载不同适配器
- **跨语言/跨模态迁移**：使用 MAD-X Adapter 实现跨语言/跨模态迁移
- **持续学习**：通过添加新适配器实现持续学习，避免灾难性遗忘

### 6.2 常见陷阱

- **过拟合小数据**：适配器参数量较少，容易在小数据集上过拟合
- **推理延迟增加**：多个适配器堆叠会导致推理延迟增加
- **适配器冲突**：不同任务的适配器可能相互冲突，影响模型性能
- **训练不稳定性**：适配器训练可能出现梯度消失或爆炸问题

### 6.3 解决方案

- **正则化**：添加 Dropout、权重衰减等正则化技术
- **适配器剪枝**：只保留性能最好的适配器
- **适配器初始化**：使用合适的初始化方法（如 Xavier 初始化）
- **学习率调整**：为适配器设置单独的学习率，通常高于主模型

## 7. 前沿演进与未来方向

### 7.1 Adapter 与 MoE 架构结合

- **核心思想**：将 Adapter 与混合专家（MoE）架构结合，实现更高效的模型适应
- **优势**：
  - 可以为每个专家添加特定任务的适配器
  - 支持动态路由到不同的适配器专家
  - 进一步提高模型的参数效率

### 7.2 模块化智能体中的 Adapter

- **应用**：在模块化智能体中，每个模块可以对应一个适配器
- **优势**：
  - 便于添加或替换智能体的功能模块
  - 支持动态组合不同的功能模块
  - 实现智能体的持续进化

### 7.3 持续学习中的 Adapter

- **核心思想**：通过为每个新任务添加适配器，实现持续学习
- **优势**：避免灾难性遗忘，因为新任务不会修改旧任务的适配器
- **最新进展**：2025 年 ICLR 论文《Continual Adapter Learning with Dynamic Task Modeling》提出了动态任务建模的持续适配器学习方法

### 7.4 近 2-3 年顶会研究进展

- **2024 NeurIPS**：《AdapterHub 2.0: A Framework for Modular and Efficient Transfer Learning》扩展了 AdapterHub 框架，支持更多模型架构和任务
- **2025 ICLR**：《Compacter V2: Efficient Adapter Design with Shared Parameters and Low-Rank Decomposition》进一步优化了 Compacter 适配器的设计
- **2025 ACL**：《Cross-Modal Adapter Fusion for Multimodal Large Language Models》提出了跨模态适配器融合方法，提高了多模态大模型的性能

## 8. 最佳实践

### 8.1 适配器设计

- **适配器大小**：通常设置为主模型隐藏维度的 1/16 到 1/4
- **插入位置**：对于大多数任务，推荐在 FFN 层后插入适配器
- **激活函数**：GELU 通常比 ReLU 表现更好
- **归一化**：在适配器前后添加层归一化有助于稳定训练

### 8.2 训练技巧

- **学习率**：适配器学习率通常设置为 1e-4 到 5e-4，高于主模型
- **训练轮次**：适配器训练轮次通常比全参数微调少，推荐 3-10 轮
- **批次大小**：根据 GPU 内存调整，通常为 4-32
- **梯度裁剪**：使用梯度裁剪防止梯度爆炸，推荐裁剪阈值为 1.0

### 8.3 部署策略

- **适配器合并**：推理时可以将适配器权重合并到主模型中，减少推理延迟
- **动态加载**：在模型服务中，可以根据需要动态加载不同的适配器
- **多适配器管理**：使用高效的适配器存储和管理机制，支持快速切换

## 9. 任务清单

- [ ] 补充 Adapter 与其他 PEFT 方法的详细对比实验
- [ ] 添加 Adapter 架构的 Mermaid 结构图
- [ ] 补充更多 Adapter 变体的实现示例
- [ ] 添加 Adapter 在实际项目中的应用案例
- [ ] 更新最新的 Adapter 研究进展

## 10. 关联内容

- 返回 [[../参数高效微调|参数高效微调]] - 参数高效微调主页面
- 参考 [[../LoRA/LoRA|LoRA]] - LoRA 详细内容
- 参考 [[../QLoRA/QLoRA|QLoRA]] - QLoRA 详细内容
- 参考 [[../../3.机器学习/微调/微调|微调]] - 微调详细内容
- 参考 [[../../5.大语言模型核心/大语言模型/大语言模型|大语言模型]] - 大语言模型详细内容

---

*本指南基于 2026 年 1 月的最新研究编写，内容将持续更新。*