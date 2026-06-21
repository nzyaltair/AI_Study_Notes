title: DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition
authors: Z.Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z.F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, Chong Ruan
year: 2025
venue: 未知
tags: [Paper, AI, Theorem Proving, Reinforcement Learning, Mathematical Reasoning]

# 📄 论文基本信息
- 标题：DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition
- 作者：Z.Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z.F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, Chong Ruan
- 发表年份：2025
- 期刊 / 会议：未知
- DOI / URL：https://github.com/deepseek-ai/DeepSeek-Prover-V2
- 本地路径：笔记\AI笔记\16.论文精读与笔记\DeepSeek\DeepSeek-Prover-V2\Ren 等 - DeepSeek-Prover-V2 Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal D.pdf

# 🎯 研究背景与问题
- 研究背景：
  - 大语言模型(LLMs)的推理能力在数学问题解决领域取得了革命性进展
  - 推理时扩展，特别是通过自然语言思维链推理，使得LLMs能够反思中间推理步骤，提高准确性和可解释性
  - 形式化定理证明需要严格的逻辑基础，而LLMs的自然语言推理本质上是非正式的
  - 证明助手如Lean、Isabelle和Coq在严格的逻辑基础上运行，不允许歧义、隐含假设或细节遗漏

- 现有方法的核心瓶颈：
  - 非正式的高级推理与形式化验证系统的语法严格性之间存在巨大差距
  - 如何利用非正式数学推理的优势来支持形式化定理证明
  - 如何将复杂的形式化证明分解为可管理的子任务

- 作者明确要解决的问题（Problem Statement）：
  - 如何通过子目标分解和强化学习来推进形式化数学推理
  - 如何将非正式和正式数学推理集成到一个统一的模型中
  - 如何构建一个能够在形式化定理证明中达到最先进性能的开源模型

# 🚀 核心贡献（Contributions）
- Contribution 1：
  - 提出了DeepSeek-Prover-V2，一个为Lean 4形式化定理证明设计的开源大语言模型
  - 新在于：通过DeepSeek-V3驱动的递归定理证明pipeline收集初始化数据，实现了非正式和正式数学推理的统一

- Contribution 2：
  - 设计了冷启动训练过程，通过提示DeepSeek-V3将复杂问题分解为子目标，并将解决的子目标证明合成为思维链过程
  - 新在于：将层次化子目标分解与强化学习相结合，使模型能够从非正式推理过渡到正式证明

- Contribution 3：
  - 引入了ProverBench，一个包含325个形式化问题的集合，用于丰富评估，包括最近AIME竞赛的15个问题
  - 新在于：提供了更全面的评估基准，特别是针对竞赛级别的数学问题

# 🧠 方法详解（Method）
## 整体框架
- 模型/算法整体流程说明：
  - 初始化数据收集：通过DeepSeek-V3驱动的递归定理证明pipeline收集数据
  - 冷启动训练：提示DeepSeek-V3将复杂问题分解为子目标，将解决的子目标证明合成为思维链过程
  - 强化学习：使用合成的思维链和DeepSeek-V3的逐步推理作为初始冷启动，进行强化学习
  - 模型评估：在标准基准和新引入的ProverBench上评估模型性能

- 输入 / 输出定义：
  - 输入：形式化数学问题（Lean 4格式）
  - 输出：形式化证明（Lean 4代码）

## 关键模块拆解
- 模块 1：递归定理证明pipeline
  - 功能：收集初始化数据，通过DeepSeek-V3分解复杂问题为子目标
  - 过程：递归地分解问题，解决子目标，合成证明过程

- 模块 2：冷启动训练过程
  - 功能：创建强化学习的初始起点
  - 过程：提示DeepSeek-V3分解问题，合成思维链，结合逐步推理

- 模块 3：强化学习框架
  - 功能：优化模型的形式化证明能力
  - 过程：使用冷启动数据作为初始点，通过强化学习改进模型性能

## 关键公式与直觉解释
- 公式：
  - 论文中未明确给出具体公式

- 含义：
  - 模型通过层次化子目标分解和强化学习来学习形式化证明
  - 子目标分解允许模型将复杂问题分解为可管理的子任务

- 直觉理解：
  - 类似于人类解决数学问题的方法，先将复杂问题分解为简单的子问题，解决子问题后再组合成完整的解决方案
  - 通过强化学习，模型学会从非正式推理过渡到严格的形式化证明
  - 冷启动过程帮助模型建立初始的证明能力，为后续的强化学习奠定基础

- 与传统方法对比：
  - 传统方法：要么完全依赖非正式推理（准确率低），要么完全依赖形式化系统（难以扩展）
  - 新方法：结合了两者的优势，通过子目标分解和强化学习实现了从非正式到正式的过渡

# 🧪 实验设计与结果分析
## 实验设置
- 数据集：
  - MiniF2F-test：形式化定理证明基准
  - PutnamBench：包含658个问题的数学竞赛基准
  - ProverBench：新引入的包含325个形式化问题的集合，包括最近AIME竞赛的15个问题

- Baseline：
  - DeepSeek-Prover-V2-7B
  - Kimina-Prover-Preview 72B
  - BFS-Prover 7B
  - STP 7B
  - Geodel-Prover 7B
  - DeepSeek-V3-0324（非正式推理）

- 评价指标：
  - 通过率（Pass Rate）：模型成功证明定理的比例
  - 解决问题数量：模型能够解决的问题总数

- 对照实验设计思路：
  - 比较不同模型在标准基准上的性能
  - 评估模型在新引入的ProverBench上的表现
  - 对比正式推理模型（DeepSeek-Prover-V2）与非正式推理模型（DeepSeek-V3）在AIME问题上的表现

## 核心实验结果
- 关键结果解读：
  - DeepSeek-Prover-V2-671B在MiniF2F-test上达到88.9%的通过率，显著高于其他模型
  - 在PutnamBench上，DeepSeek-Prover-V2-671B解决了658个问题中的47个，远高于其他模型
  - 在ProverBench的15个AIME问题上，DeepSeek-Prover-V2-671B成功解决了6个，而DeepSeek-V3使用多数投票解决了8个
  - 结果表明正式和非正式数学推理之间的差距正在缩小

- 是否支撑作者主张：
  - 是的，实验结果表明DeepSeek-Prover-V2在形式化定理证明中达到了最先进的性能
  - 结果支持了作者关于通过子目标分解和强化学习推进形式化数学推理的主张
  - 非正式和正式数学推理之间的差距缩小，验证了模型设计的有效性

# 🔍 讨论与局限性
- 方法优势：
  - 成功整合了非正式和正式数学推理，实现了从前者到后者的过渡
  - 通过子目标分解和强化学习，显著提高了模型的形式化证明能力
  - 开源了模型和评估基准，促进了研究社区的发展
  - 在标准基准和新引入的ProverBench上都取得了优异的性能

- 潜在局限：
  - 模型规模较大（671B参数），可能限制在资源受限环境中的部署
  - 尽管性能显著提升，但与非正式推理模型相比，在AIME问题上仍有差距
  - 模型专注于Lean 4，可能需要额外工作才能适应其他证明助手

- 隐含假设：
  - DeepSeek-V3的非正式推理能力足以指导子目标分解
  - 子目标分解是形式化定理证明的有效策略
  - 强化学习能够有效优化模型的形式化证明能力

- 实际应用风险：
  - 模型可能会生成无法通过形式化验证的证明
  - 对于非常复杂的定理，模型可能无法完成证明
  - 模型的推理过程可能缺乏可解释性

# 💡 可扩展方向 / 个人思考
- 可改进点：
  - 探索更小规模的模型架构，在保持性能的同时减少计算需求
  - 扩展模型以支持更多证明助手（如Isabelle、Coq等）
  - 进一步优化子目标分解策略，提高分解的质量和效率
  - 研究如何更好地结合非正式和正式推理，缩小两者之间的差距

- 可用于哪些研究方向：
  - 形式化定理证明
  - 数学推理
  - 强化学习在复杂任务中的应用
  - 多模态推理（非正式到正式）

- 是否值得复现 / 跟进：
  - 是的，该方法展示了一种有效的途径来推进形式化数学推理
  - 开源模型和评估基准为进一步研究提供了良好的基础
  - 结果表明该方法在形式化定理证明中具有巨大潜力

# 🔗 相关论文链接（双链）
- [[DeepSeek-V3]]
- [[Formal Theorem Proving]]
