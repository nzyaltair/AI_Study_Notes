# DPO

## 背景与发展

DPO（Direct Preference Optimization）是一种直接从偏好数据优化语言模型的对齐方法，由 Rafailov et al. (2023, Stanford) 提出。DPO 通过理论推导将奖励函数隐式地表示为策略概率比，从而跳过奖励模型训练和 PPO 优化，大幅简化对齐流程。DPO 的提出使偏好对齐从复杂的强化学习问题转变为简单的分类问题。

## 核心思想

DPO 的关键洞察是：在 RLHF 的 KL 约束优化问题中，最优策略与奖励函数之间存在闭式解关系，因此可以直接从偏好数据中学习最优策略，无需显式训练奖励模型。

## 技术原理

### Bradley-Terry 偏好模型

人类对两个输出的偏好可以用相对奖励建模：

$$P(y_w \succ y_l \mid x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

### 策略正则化推理

RLHF 中 PPO 优化的目标为：

$$J(\phi) = \mathbb{E}_{(x,y) \sim p_\phi}\left[r(x,y)\right] - \beta \, D_{KL}\left(p_\phi \| p_{ref}\right)$$

对此优化问题求解，可得最优策略的闭式解：

$$\log p_\phi^*(y|x) = \log p_{ref}(y|x) + \frac{r(x,y)}{\beta} + C(x)$$

即奖励函数可表示为策略与参考策略的对数概率比：

$$r(x,y) = \beta \log \frac{p_\phi^*(y|x)}{p_{ref}(y|x)} + C(x)$$

### DPO 损失函数

将上述关系代入 Bradley-Terry 模型，得到 DPO 损失：

$$L_{DPO}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[\log\sigma\left(\beta\left(\log\frac{p_\phi(y_w|x)}{p_{ref}(y_w|x)} - \log\frac{p_\phi(y_l|x)}{p_{ref}(y_l|x)}\right)\right)\right]$$

其中：
- $y_w$ 是偏好输出（chosen），$y_l$ 是非偏好输出（rejected）
- $p_\phi$ 是当前策略，$p_{ref}$ 是参考策略（SFT 模型）
- $\beta$ 是温度参数，控制策略偏离参考策略的程度

### 参考策略的作用

参考策略 $p_{ref}$（通常为 SFT 模型）在 DPO 中起到正则化作用：防止模型过度拟合偏好数据，确保模型输出与初始模型保持一致性。

### 关键超参数

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| 学习率 | 1e-6 ~ 5e-5 | 通常比 SFT 略高 |
| $\beta$ | 0.1 ~ 1.0 | 小模型 0.3-1.0，大模型 0.1-0.3 |
| Batch size | 32 ~ 128 | 越大越稳定 |
| 训练轮次 | 3 ~ 10 | 通常比 RLHF 少 |

$\beta$ 过小：模型过度偏离参考策略，过拟合。$\beta$ 过大：模型难以学习偏好，性能提升有限。

### DPO 与 RLHF 对比

| 维度 | RLHF | DPO |
|------|------|-----|
| 训练阶段 | 4（数据+SFT+RM+PPO） | 2（数据+SFT→DPO） |
| 计算成本 | 高 | 约为 RLHF 的 1/3 |
| 训练稳定性 | 低（PPO 不稳定） | 高（分类问题） |
| 超参数 | 多（KL、clip、lr 等） | 少（主要调 $\beta$） |

## 发展演进

DPO (2023, Stanford) → IPO (2023) → KTO (2023) → SimPO (2024，去除参考模型) → Iterative DPO (2024) → 多模态 DPO

## 关键算法·模型

- **DPO**：直接偏好优化，将 RL 转化为分类
- **IPO**：扩展 DPO 到更一般的偏好数据格式
- **KTO**：结合知识蒸馏，保留初始模型知识
- **SimPO**：去除参考策略项，简化计算
- **ORPO**：在 SFT 阶段同时进行偏好优化

## 应用场景

- **资源受限环境**：计算成本低，适合资源有限场景
- **大规模模型微调**：训练稳定，适合大模型
- **快速迭代**：流程简单，适合快速实验
- **多任务偏好对齐**：可扩展到多任务场景

## 与其他技术关系

- DPO 是 [[RLHF]] 的简化方案，数学上等价但流程更简
- DPO 以 [[监督微调]] 模型作为参考策略
- [[RLAIF]] 的 AI 反馈数据可直接用于 DPO 训练
- SimPO、IPO、KTO 等是 DPO 的改进变体
