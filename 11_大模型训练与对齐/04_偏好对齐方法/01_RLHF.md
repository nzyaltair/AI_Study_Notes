# RLHF

## 背景与发展

RLHF（Reinforcement Learning from Human Feedback）是通过人类反馈训练和优化大语言模型的对齐方法。Christiano et al. (2017) 提出 RLHF 的雏形，InstructGPT (2022) 将其系统化为三阶段流程，ChatGPT 将 RLHF 推向主流。RLHF 是目前最成熟、应用最广泛的偏好对齐技术。

## 核心思想

通过收集人类对模型输出的偏好排序数据，训练一个奖励模型（Reward Model）来预测人类偏好，再使用强化学习算法（PPO）以奖励模型的输出为指导优化语言模型，使模型输出与人类偏好对齐。

## 技术原理

### 三阶段流程

1. **监督微调（SFT）**：训练一个初始的指令跟随模型，作为 RLHF 的起点
2. **训练奖励模型（RM）**：在 SFT 模型基础上，用人类偏好数据训练奖励模型
3. **PPO 优化**：以奖励模型输出为奖励信号，用强化学习优化 SFT 模型

### 奖励模型训练

对于输入 $x$ 和输出 $y$，奖励模型 $r_\theta(x, y)$ 预测人类偏好得分。使用 Bradley-Terry 模型，训练目标为最大化偏好输出与非偏好输出得分差：

$$L_{RM}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[\log\sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

其中 $y_w$ 是人类偏好的输出（chosen），$y_l$ 是不偏好的输出（rejected），$\sigma$ 是 sigmoid 函数。

### PPO 优化目标

PPO 优化的目标是最大化奖励期望，同时通过 KL 散度约束防止模型偏离初始策略太远：

$$J(\phi) = \mathbb{E}_{(x, y) \sim p_\phi} \left[r_\theta(x, y)\right] - \beta \, D_{KL}\left(p_\phi(y|x) \| p_{\phi_0}(y|x)\right)$$

其中：
- $p_\phi$ 是当前策略（正在优化的模型）
- $p_{\phi_0}$ 是参考策略（SFT 模型，固定不变）
- $\beta$ 是 KL 散度惩罚系数，控制策略偏离程度

### PPO 核心组件

- **策略网络（Actor）**：需要优化的语言模型
- **价值网络（Critic）**：估计状态值，减少梯度方差
- **奖励模型**：提供外部奖励信号
- **参考模型**：SFT 模型，用于计算 KL 约束

### 关键超参数

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| 学习率 | 1e-6 ~ 5e-5 | 低于 SFT 学习率 |
| KL 系数 $\beta$ | 0.1 ~ 1.0 | 控制策略偏离程度 |
| PPO clip range | 0.1 ~ 0.3 | 裁剪范围，限制策略更新幅度 |
| Batch size | 32 ~ 256 | 越大越稳定 |

## 发展演进

Christiano et al. (2017，RLHF 雏形) → InstructGPT (2022，三阶段标准化) → ChatGPT (2022，大规模 RLHF) → Iterative RLHF (2023，迭代优化) → 多模态 RLHF (2024)

## 关键算法·模型

- **Bradley-Terry 模型**：偏好建模的经典模型
- **PPO**：Proximal Policy Optimization，策略裁剪的强化学习算法
- **InstructGPT**：RLHF 三阶段流程的标杆
- **ChatGPT**：RLHF 的大规模工业应用
- **Constitutional AI**：Anthropic 提出的结合规则的 RLHF 变体

## 应用场景

- **对话系统**：训练安全、友好、有用的对话模型
- **内容生成**：生成符合人类偏好的文本和代码
- **搜索引擎**：优化搜索结果排序
- **安全对齐**：防止有害输出

## 与其他技术关系

- RLHF 在 [[监督微调]] 之后进行，SFT 模型是 RLHF 的起点
- [[DPO]] 是 RLHF 的简化方案，去除奖励模型和 PPO
- [[RLAIF]] 用 AI 反馈替代 RLHF 中的人类反馈
- [[GRPO]] 去除奖励模型和 Critic，用组内比较替代
- KL 约束中的散度计算是信息论的基础工具
