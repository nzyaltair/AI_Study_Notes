# LLM训练与对齐

## 背景

LLM的训练是一个多阶段流程：预训练→监督微调（SFT）→偏好对齐（RLHF/DPO）。这三个阶段分别赋予模型语言能力、指令跟随能力和人类偏好对齐，是ChatGPT等现代AI助手的核心训练范式。

## 核心思想

- **预训练**：在海量文本上学习通用语言表示（"读万卷书"）
- **监督微调**：用指令-回答对教会模型跟随指令（"学规矩"）
- **偏好对齐**：通过人类反馈使模型输出符合偏好（"学做人"）

## 技术原理

### 预训练

- **目标**：Next Token Prediction（自回归语言建模）
  - $L = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$
- **数据**：万亿级token的网页文本、书籍、代码
- **训练**：分布式训练（数据并行/张量并行/流水线并行/ZeRO）
- **关键超参**：学习率（Warmup + Cosine）、Batch Size、序列长度

### 监督微调（SFT）

- **目标**：让模型学会跟随指令格式回答问题
- **数据**：高质量指令-回答对（Alpaca/ShareGPT/人工标注）
- **训练**：
  - 全量微调（小模型）
  - 参数高效微调（大模型，LoRA/QLoRA）
- **关键**：数据质量 > 数据数量，"Less is More"

### 参数高效微调（PEFT）

| 方法 | 原理 | 参数量 | 效果 |
|------|------|--------|------|
| **LoRA** | 冻结原始权重，训练低秩矩阵 $W + BA$ | ~1% | 接近全量 |
| **QLoRA** | 4-bit量化基模 + LoRA | ~0.1% | 接近全量 |
| **Adapter** | 在层间插入小模块 | ~3% | 略差于LoRA |
| **Prefix Tuning** | 学习虚拟前缀嵌入 | ~0.1% | 略差于LoRA |

**LoRA核心公式**：$W' = W + \Delta W = W + BA$，其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$

### 偏好对齐

#### RLHF（Reinforcement Learning from Human Feedback）

三阶段流程：
1. **SFT模型**：作为初始策略
2. **训练奖励模型（RM）**：用人类偏好数据训练
   - $L_{RM} = -\log\sigma(r(x, y_w) - r(x, y_l))$（w=chosen, l=rejected）
3. **PPO优化**：用RM作为奖励，强化学习优化SFT模型
   - $\max_\pi \mathbb{E}[r(x,y)] - \beta \text{KL}(\pi || \pi_{SFT})$

#### DPO（Direct Preference Optimization）

无需训练奖励模型，直接从偏好数据优化策略：

$$L_{DPO} = -\log\sigma\left(\beta\log\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

- **优势**：简化训练流程，稳定性更好
- **变体**：IPO、KTO、ORPO、SimPO

#### 其他对齐方法

| 方法 | 特点 |
|------|------|
| **RLAIF** | 用AI反馈代替人类反馈 |
| **Constitutional AI** | 基于宪法的自我修正 |
| **Red Teaming** | 红队对抗，发现漏洞 |

### 数据处理

- **预训练数据**：去重（MinHash）、质量过滤、有害内容过滤
- **SFT数据**：指令多样性、难度分布、拒绝采样
- **偏好数据**：人类标注/AI生成、Pair-wise比较

## 发展演进

GPT-1（预训练+微调）→ GPT-2（零样本）→ GPT-3（上下文学习）→ InstructGPT（SFT+RLHF）→ ChatGPT（RLHF+安全对齐）→ DPO（2023，简化对齐）→ 推理模型o1（2024，强化学习+CoT）

## 应用领域

- **基座模型训练**：LLaMA/Qwen/DeepSeek的预训练
- **领域微调**：医疗、法律、金融等领域适配
- **对齐安全**：防止有害输出、注入价值观
- **个性化**：基于用户偏好微调

## 与其他技术关系

- 预训练基于[[Transformer架构详解|Transformer]]和[[预训练语言模型|预训练范式]]
- 分布式训练依赖[[训练技巧与稳定性|训练技巧]]
- LoRA等PEFT方法是[[大模型相关数学|低秩近似]]的应用
- 对齐后的模型用于[[Agent智能体|Agent]]和[[多模态模型|多模态]]应用
- 推理优化详见[[LLM推理与优化]]
