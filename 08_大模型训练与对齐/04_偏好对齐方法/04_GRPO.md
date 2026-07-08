# GRPO

## 背景与发展

GRPO（Group Relative Policy Optimization）是 DeepSeek 提出的强化学习对齐算法，在 DeepSeek-R1 中首次大规模应用。它简化了传统 RLHF 的流程，去除了独立的奖励模型和价值网络（Critic），直接使用组内相对优势作为奖励信号，显著降低训练成本和复杂度。GRPO 的成功证明了无需复杂 RLHF 流程也能训练出高性能推理模型。

## 核心思想

对同一问题生成一组（Group）候选回答，通过组内相对比较计算每个回答的优势值（advantage），无需训练独立的奖励模型，也无需 Critic 网络。使用 PPO 变体优化策略，使优质回答概率提升、劣质回答概率降低。

## 技术原理

### 算法流程

1. 输入问题 $q$，从当前策略 $\pi_\theta$ 采样 $G$ 个回答 $\{o_1, o_2, ..., o_G\}$
2. 计算每个回答的奖励 $r_i$（规则奖励或模型奖励）
3. 组内标准化计算优势值：$A_i = (r_i - \text{mean}(r)) / \text{std}(r)$
4. 用 PPO 更新策略 $\pi_\theta$，最大化 $\sum_i A_i \cdot \log\pi_\theta(o_i|q)$

### 优势值计算

组内标准化将绝对奖励转化为相对优势：

$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

这种方法的关键在于：不需要绝对奖励值精确，只需要组内相对排序正确。

### 策略优化目标

带 KL 惩罚和 PPO 裁剪的优化目标：

$$J_{GRPO} = \mathbb{E}\left[\sum_i A_i \cdot \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)}, \, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)}, 1-\epsilon, 1+\epsilon\right)\right)\right] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

### 与 RLHF 的关键区别

| 特征 | RLHF（PPO） | GRPO |
|------|------------|------|
| 奖励模型 | 需要独立训练 | 不需要（组内比较） |
| 基线计算 | 价值网络（Critic） | 组内均值 |
| 训练成本 | 高（RM + Critic + Actor） | 低（仅 Actor） |
| 显存占用 | 大（3 个模型） | 小（1 个模型） |
| 稳定性 | Critic 训练不稳定 | 更稳定（无 Critic） |
| 适用场景 | 通用对齐 | 推理任务（有明确答案） |

### 奖励函数设计

DeepSeek-R1 使用两类奖励：

- **正确性奖励**：数学题答案正确性、代码通过测试用例
- **格式奖励**：思维链使用 `<think>` 标签包裹

这种基于规则的奖励避免了奖励模型的偏差，特别适合有明确正确答案的推理任务。

## 发展演进

PPO (2017) → RLHF-PPO (2022, ChatGPT) → DPO (2023，去除 RM 但仍为监督学习) → GRPO (2024，去除 RM + Critic，组内相对优化) → DeepSeek-R1 (2025，证明 GRPO 可训练推理模型)

## 关键算法·模型

- **GRPO**：组内相对策略优化
- **DeepSeek-R1**：使用 GRPO 训练的推理模型，性能对标 OpenAI o1
- **DeepSeek-R1-Zero**：直接在基座模型上用 GRPO 训练，无 SFT 阶段

## 应用场景

- **推理模型训练**：数学推理、代码生成等有明确正确答案的任务
- **低成本对齐**：无需训练奖励模型，适合资源受限场景
- **大规模训练**：显存占用小，适合大规模分布式训练

## 与其他技术关系

- GRPO 是 [[RLHF]] 的简化变体，去除了奖励模型和 Critic
- 与 [[DPO]] 不同，GRPO 仍使用在线强化学习（on-policy）
- GRPO 是推理模型训练的关键技术
- KL 惩罚项参考了 RLHF 的设计
