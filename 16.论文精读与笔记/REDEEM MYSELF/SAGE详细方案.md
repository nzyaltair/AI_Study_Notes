
### _A. Design Rationale_


神经网络的解释性研究表明，浅层主要提取宏观特征（全局结构信息），中层和深层主要提取微观特征（细粒度细节）[28]。为了保证隐蔽性，触发器通常设计为对样本全局结构影响较小的微扰动，因此主要影响深层而不影响浅层。神经注意力转移（NAD）从带有后门的教师模型中微调学生模型，因此教师模型可能具有好的浅层但坏的深层（可能略优于学生模型的深层）。NAD 使学生模型的好浅层从教师模型的好浅层学习，学生模型的坏深层从教师模型的坏深层学习。这样，教师模型的深层不能有效地纠正学生模型的深层。相比之下，我们采用自注意力蒸馏（SAD），通过模型本身的注意力图对齐，让不好的深层从好的浅层中学习，从而潜在地纠正深层。


为了实现设计目标，SAGE 执行三个关键步骤，即注意力表示、损失计算和学习率更新，如图 1 所示。在注意力表示模块中，我们根据每个神经元对预测结果的贡献提取其注意力。在损失计算模块中，我们根据浅层的注意力知识纠正深层的权重，同时保持模型预测精度。在学习率更新模块中，我们没有使用现有的自适应方法（每 2 个时期将学习率除以 10 [32]），而是设计了一种新颖的学习率自适应策略，该策略会仔细跟踪干净样本的预测精度以指导学习率调整。


### _B. Attention Representation_


注意力表示对于成功纠正后门至关重要，因为我们希望提取必要的注意力知识来指导自我净化过程。给定一个带后门的模型 FB，第 l 层的激活张量表示为$ FB^l$ ∈ $R^{Cl×Hl×Wl}$，其中 Cl、Hl 和 Wl 分别表示第 l 层注意力图的通道维度、高度和宽度。注意力表示相当于寻求一个映射函数 G：$R^{Cl×Hl×Wl} → R^{Hl×Wl}$。在图中，每个元素的绝对值表示其对最终分类结果的贡献。我们通过计算所有通道上这些值的统计数据来构建映射函数 G。如 [22]、[32] 中所建议的，映射函数可以采用以下四种形式之一。


![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2e5fd51b-178a-81f9-8e9d-0003e80f0e8a/81cdaf61-2691-49eb-99a0-59ba64c55b66/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=ASIAZI2LB466QTIZSIVJ%2F20260114%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260114T101004Z&X-Amz-Expires=3600&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFIaCXVzLXdlc3QtMiJHMEUCIF6Fg4m7ugV3JR2Hf8pZ6FNP8WW6jD2RHnsWQkNNNsBgAiEA%2FXMARq3%2FsULPY57xoh3L1uv2c8LnGq1e3F8DCXLUKEkq%2FwMIGxAAGgw2Mzc0MjMxODM4MDUiDKpxlQkSrq6A1fSuHyrcA%2FYXCO9EyasxETnhl5qHcnRnPnDZPAScBET8bmMt3oUo5yZe34fR116TeOJZkRRYzbNCq%2FUsgUz18hNw19tF%2Fxj8ZLjKbnPznhhk2oYEThoB%2F%2BIsnPj1wOOcoWgMVdOXPc3Sah6p%2B8Ni5EZjV0C5A8j%2BisC7HP8roEIOplqLhNU6ohgcRvv9PQ%2BqhWU4wrg5Jen0rRezEm98cTAFQYeMoVperr8i47No02MX1Hv5CZhkBMIirDf3LvrHwjA8ghcXJgidimURWCbbkmyen52ptiJKvifnOropACJISaNZszCLKJhZEskjrvlM5PvfipMhng%2FR%2FtsgtHFCCUY4g25a%2FTP5OCA%2Bx0q0IMFPrNzjom1cdnVtNMGbOo1H4B4qMXJIjzQDtatuPbN96ezwVfwDbMsWsSHNy1Ye4UwnmJfnnDatFfTN6Qaidls3nLypDKQNmwxHZfpLZzTBb%2B4ubUwLSfq2qZKRAJ904p2CVWyUixFE9r1AgiH84ZV5aieDSvDrAVmWFu93dArPbz5vno0KicPiDmStgmb8MAcUzDlRgPHJwf4BkPWPd4kL447gaiMJSw2I0t2LdCr%2BNUCMaRNKqtWH4GPQAqQImGrpimm%2FJgsfLAzEdqTERHefBbFHMNvRncsGOqUBU9NZARDrcipsG3LbiOAWpGr0LQhhxFlw0fPyIemmqQLP8uIyAw7cGdBBPpUIZldoXlN%2F3VE0XtXDxMF3gtPxGIoC%2BuLZMVnhbbfZuRS30CfJDumzFAiGCMITnxlXI09gnb76NoH3fnN1VqclHn7DmmNtnsGV7IGqQdsD7%2BkI8nxyQkC2G68mwW1IRCoKJf%2Bt98RT%2F7ayp%2Bqedf0m2RuCa3du2MJA&X-Amz-Signature=56aa35c85850bd1ef2abdffe80d3026d6f0c43102a01c142fa7d19601d024efe&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject)


其中 $FB^l$i是第 l 层的 $FB^l$的第 i 个切片，p > 1是常数参数。


$G_{sum}^p$是 $G_{sum} $的升级版，它放大了激活度较高的权重。p 越大，激活度较高的区域获得的权重就越高。与仅选择最大神经元激活度作为权重的 $G_{max}^p$不同，$G_{sum}^p$和 $G_{mean}^p$考虑了所有区域的权重。因此，我们采用$ G_{sum}^p$(p ≥ 1) 和$ G_{mean}^p$(p ≥ 1)。此外，我们研究了哪种映射函数在评估中实现了最佳防御性能。


### _C. Loss Calculation_


提炼出的注意力知识用于指导模型的净化。与在教师模型和学生模型 [28] 中进行的传统注意力提炼不同，我们提出的自注意力提炼 (SAD) 是在单个模型中通过自上而下和分层的方式进行的。关键思想是利用浅层的注意力图作为深层注意力图的一种监督形式。自注意力提炼损失 LSAD 表示为


![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2e5fd51b-178a-81f9-8e9d-0003e80f0e8a/3638ff92-a501-468f-8302-d3d2197a4b01/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=ASIAZI2LB466QTIZSIVJ%2F20260114%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260114T101004Z&X-Amz-Expires=3600&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFIaCXVzLXdlc3QtMiJHMEUCIF6Fg4m7ugV3JR2Hf8pZ6FNP8WW6jD2RHnsWQkNNNsBgAiEA%2FXMARq3%2FsULPY57xoh3L1uv2c8LnGq1e3F8DCXLUKEkq%2FwMIGxAAGgw2Mzc0MjMxODM4MDUiDKpxlQkSrq6A1fSuHyrcA%2FYXCO9EyasxETnhl5qHcnRnPnDZPAScBET8bmMt3oUo5yZe34fR116TeOJZkRRYzbNCq%2FUsgUz18hNw19tF%2Fxj8ZLjKbnPznhhk2oYEThoB%2F%2BIsnPj1wOOcoWgMVdOXPc3Sah6p%2B8Ni5EZjV0C5A8j%2BisC7HP8roEIOplqLhNU6ohgcRvv9PQ%2BqhWU4wrg5Jen0rRezEm98cTAFQYeMoVperr8i47No02MX1Hv5CZhkBMIirDf3LvrHwjA8ghcXJgidimURWCbbkmyen52ptiJKvifnOropACJISaNZszCLKJhZEskjrvlM5PvfipMhng%2FR%2FtsgtHFCCUY4g25a%2FTP5OCA%2Bx0q0IMFPrNzjom1cdnVtNMGbOo1H4B4qMXJIjzQDtatuPbN96ezwVfwDbMsWsSHNy1Ye4UwnmJfnnDatFfTN6Qaidls3nLypDKQNmwxHZfpLZzTBb%2B4ubUwLSfq2qZKRAJ904p2CVWyUixFE9r1AgiH84ZV5aieDSvDrAVmWFu93dArPbz5vno0KicPiDmStgmb8MAcUzDlRgPHJwf4BkPWPd4kL447gaiMJSw2I0t2LdCr%2BNUCMaRNKqtWH4GPQAqQImGrpimm%2FJgsfLAzEdqTERHefBbFHMNvRncsGOqUBU9NZARDrcipsG3LbiOAWpGr0LQhhxFlw0fPyIemmqQLP8uIyAw7cGdBBPpUIZldoXlN%2F3VE0XtXDxMF3gtPxGIoC%2BuLZMVnhbbfZuRS30CfJDumzFAiGCMITnxlXI09gnb76NoH3fnN1VqclHn7DmmNtnsGV7IGqQdsD7%2BkI8nxyQkC2G68mwW1IRCoKJf%2Bt98RT%2F7ayp%2Bqedf0m2RuCa3du2MJA&X-Amz-Signature=17246db54e9e9632936c1d68101aa29a87c6ea8349bb54ce45d6fb0bf0887cd7&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject)


其中 G(·) 用于提取特征表示，U 表示将浅层和深层的注意力表示统一为相同大小的双线性上采样操作，ψ1 和 ψ2 表示在两个不同层生成的注意力图的归一化向量，Ld(·) 表示两个注意力图之间的距离，我们使用 L2 正则来计算距离。我们在映射函数上应用归一化函数 N(·)，而不是 softmax 函数 [22]。归一化操作对于我们的 SAD 至关重要，因为 SAD 跨阴影层和深层执行注意力图对齐，但阴影层和深层的注意力图具有不同的绝对值。如果没有归一化，阴影层和深层的注意力图对齐将引起扭曲，从而可能降低干净样本的预测精度。


重新训练时只使用SAD损失LSAD会大大降低干净样本的预测精度。这是因为自注意力蒸馏仅仅是为了最小化不同层的注意力图之间的差异，而不管对齐的注意力图是否符合正确的预测结果。为了解决这个问题，我们添加了交叉熵（CE）损失$L_{CE}$来保证干净样本的预测结果的精度。


![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2e5fd51b-178a-81f9-8e9d-0003e80f0e8a/54e11a93-c815-4c79-84fe-847156e7545f/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=ASIAZI2LB466QTIZSIVJ%2F20260114%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260114T101004Z&X-Amz-Expires=3600&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFIaCXVzLXdlc3QtMiJHMEUCIF6Fg4m7ugV3JR2Hf8pZ6FNP8WW6jD2RHnsWQkNNNsBgAiEA%2FXMARq3%2FsULPY57xoh3L1uv2c8LnGq1e3F8DCXLUKEkq%2FwMIGxAAGgw2Mzc0MjMxODM4MDUiDKpxlQkSrq6A1fSuHyrcA%2FYXCO9EyasxETnhl5qHcnRnPnDZPAScBET8bmMt3oUo5yZe34fR116TeOJZkRRYzbNCq%2FUsgUz18hNw19tF%2Fxj8ZLjKbnPznhhk2oYEThoB%2F%2BIsnPj1wOOcoWgMVdOXPc3Sah6p%2B8Ni5EZjV0C5A8j%2BisC7HP8roEIOplqLhNU6ohgcRvv9PQ%2BqhWU4wrg5Jen0rRezEm98cTAFQYeMoVperr8i47No02MX1Hv5CZhkBMIirDf3LvrHwjA8ghcXJgidimURWCbbkmyen52ptiJKvifnOropACJISaNZszCLKJhZEskjrvlM5PvfipMhng%2FR%2FtsgtHFCCUY4g25a%2FTP5OCA%2Bx0q0IMFPrNzjom1cdnVtNMGbOo1H4B4qMXJIjzQDtatuPbN96ezwVfwDbMsWsSHNy1Ye4UwnmJfnnDatFfTN6Qaidls3nLypDKQNmwxHZfpLZzTBb%2B4ubUwLSfq2qZKRAJ904p2CVWyUixFE9r1AgiH84ZV5aieDSvDrAVmWFu93dArPbz5vno0KicPiDmStgmb8MAcUzDlRgPHJwf4BkPWPd4kL447gaiMJSw2I0t2LdCr%2BNUCMaRNKqtWH4GPQAqQImGrpimm%2FJgsfLAzEdqTERHefBbFHMNvRncsGOqUBU9NZARDrcipsG3LbiOAWpGr0LQhhxFlw0fPyIemmqQLP8uIyAw7cGdBBPpUIZldoXlN%2F3VE0XtXDxMF3gtPxGIoC%2BuLZMVnhbbfZuRS30CfJDumzFAiGCMITnxlXI09gnb76NoH3fnN1VqclHn7DmmNtnsGV7IGqQdsD7%2BkI8nxyQkC2G68mwW1IRCoKJf%2Bt98RT%2F7ayp%2Bqedf0m2RuCa3du2MJA&X-Amz-Signature=d290e4e184a5a91ea2fb6670ad224730159f89ed13eeedf8fc46ec80b8fa58bf&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject)


其中 LCE 测量净化模型在干净样本上的预测精度，βk 是平衡自注意力蒸馏和预测精度保存的超参数。我们稍后将评估 βk 对 SAGE 性能的影响。


**算法1 自注意力蒸馏过程算法**


![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2e5fd51b-178a-81f9-8e9d-0003e80f0e8a/ac4ba185-e5cd-423d-9a5f-130e8d9bda8f/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=ASIAZI2LB466QTIZSIVJ%2F20260114%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260114T101004Z&X-Amz-Expires=3600&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFIaCXVzLXdlc3QtMiJHMEUCIF6Fg4m7ugV3JR2Hf8pZ6FNP8WW6jD2RHnsWQkNNNsBgAiEA%2FXMARq3%2FsULPY57xoh3L1uv2c8LnGq1e3F8DCXLUKEkq%2FwMIGxAAGgw2Mzc0MjMxODM4MDUiDKpxlQkSrq6A1fSuHyrcA%2FYXCO9EyasxETnhl5qHcnRnPnDZPAScBET8bmMt3oUo5yZe34fR116TeOJZkRRYzbNCq%2FUsgUz18hNw19tF%2Fxj8ZLjKbnPznhhk2oYEThoB%2F%2BIsnPj1wOOcoWgMVdOXPc3Sah6p%2B8Ni5EZjV0C5A8j%2BisC7HP8roEIOplqLhNU6ohgcRvv9PQ%2BqhWU4wrg5Jen0rRezEm98cTAFQYeMoVperr8i47No02MX1Hv5CZhkBMIirDf3LvrHwjA8ghcXJgidimURWCbbkmyen52ptiJKvifnOropACJISaNZszCLKJhZEskjrvlM5PvfipMhng%2FR%2FtsgtHFCCUY4g25a%2FTP5OCA%2Bx0q0IMFPrNzjom1cdnVtNMGbOo1H4B4qMXJIjzQDtatuPbN96ezwVfwDbMsWsSHNy1Ye4UwnmJfnnDatFfTN6Qaidls3nLypDKQNmwxHZfpLZzTBb%2B4ubUwLSfq2qZKRAJ904p2CVWyUixFE9r1AgiH84ZV5aieDSvDrAVmWFu93dArPbz5vno0KicPiDmStgmb8MAcUzDlRgPHJwf4BkPWPd4kL447gaiMJSw2I0t2LdCr%2BNUCMaRNKqtWH4GPQAqQImGrpimm%2FJgsfLAzEdqTERHefBbFHMNvRncsGOqUBU9NZARDrcipsG3LbiOAWpGr0LQhhxFlw0fPyIemmqQLP8uIyAw7cGdBBPpUIZldoXlN%2F3VE0XtXDxMF3gtPxGIoC%2BuLZMVnhbbfZuRS30CfJDumzFAiGCMITnxlXI09gnb76NoH3fnN1VqclHn7DmmNtnsGV7IGqQdsD7%2BkI8nxyQkC2G68mwW1IRCoKJf%2Bt98RT%2F7ayp%2Bqedf0m2RuCa3du2MJA&X-Amz-Signature=a31a59ee3f66dd174f12152a239f2bfe181440f81197a6de9d91cda0cfe11139&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject)


输入：后门模型 FB，一小组良性样本$S = {(x^{(t)}, y^{(t)})}_{t=1}^m$，初始学习率 η0，预设检查点 W = {$w_0, w_1, ..., w_n$}，epochs，SAD路径paths
结果：纯化后的 FB


SAD 损失是在模型中的每一对层（称为路径）上计算的。例如，给定一个 4 层模型，SAD 损失是在每对相邻层之间计算的，即路径 = {< 1, 2 >, < 2, 3 >, < 3, 4 >}。请注意，可以添加额外的路径，例如，{< 1, 3 >}、{< 2, 4 >} 和 {< 1, 4 >}。l 层模型中可能路径的数量高达 l(l−1)/2。我们将探讨哪些 SAD 路径在评估中产生最佳防御性能


_**D. Learning Rate Update**_


学习率是优化算法中一个关键的可配置超参数。学习率通常取一个较小的正实值，范围从 0 到 1。它控制每次迭代的步长，以保证稳定收敛到损失函数的最小值。设置合适的学习率并非易事。较大的学习率会导致每次迭代中的权重发生显著变化，从而导致训练期间性能不稳定。较小的学习率会导致收敛速度慢，并可能陷入局部最优解[4]。


传统的注意力蒸馏策略要么使用固定的学习率 [22]，要么启发式地调整学习率（例如，每 2 个时期将学习率除以 10）[32]。由于 SAGE 试图同时实现后门消除和预测精度保持的目标，因此学习率在控制蒸馏过程中起着重要作用。因此，我们提出了一种新策略，该策略会仔细跟踪干净样本的预测精度以指导学习率调整。


我们指定了两个降低学习率的条件。第一个条件 C1 是干净数据的损失在某个间隔内的特定时期内不会下降。第二个条件 C2 是干净数据的最大损失在某个间隔内不会下降（保持不变）。在上述情况之一中，我们将学习率除以 2，因为需要抑制蒸馏过程以限制干净数据预测精度的下降。


![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2e5fd51b-178a-81f9-8e9d-0003e80f0e8a/a2b14f86-4108-4570-8c31-6188d2cd4e48/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=ASIAZI2LB466QTIZSIVJ%2F20260114%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260114T101004Z&X-Amz-Expires=3600&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFIaCXVzLXdlc3QtMiJHMEUCIF6Fg4m7ugV3JR2Hf8pZ6FNP8WW6jD2RHnsWQkNNNsBgAiEA%2FXMARq3%2FsULPY57xoh3L1uv2c8LnGq1e3F8DCXLUKEkq%2FwMIGxAAGgw2Mzc0MjMxODM4MDUiDKpxlQkSrq6A1fSuHyrcA%2FYXCO9EyasxETnhl5qHcnRnPnDZPAScBET8bmMt3oUo5yZe34fR116TeOJZkRRYzbNCq%2FUsgUz18hNw19tF%2Fxj8ZLjKbnPznhhk2oYEThoB%2F%2BIsnPj1wOOcoWgMVdOXPc3Sah6p%2B8Ni5EZjV0C5A8j%2BisC7HP8roEIOplqLhNU6ohgcRvv9PQ%2BqhWU4wrg5Jen0rRezEm98cTAFQYeMoVperr8i47No02MX1Hv5CZhkBMIirDf3LvrHwjA8ghcXJgidimURWCbbkmyen52ptiJKvifnOropACJISaNZszCLKJhZEskjrvlM5PvfipMhng%2FR%2FtsgtHFCCUY4g25a%2FTP5OCA%2Bx0q0IMFPrNzjom1cdnVtNMGbOo1H4B4qMXJIjzQDtatuPbN96ezwVfwDbMsWsSHNy1Ye4UwnmJfnnDatFfTN6Qaidls3nLypDKQNmwxHZfpLZzTBb%2B4ubUwLSfq2qZKRAJ904p2CVWyUixFE9r1AgiH84ZV5aieDSvDrAVmWFu93dArPbz5vno0KicPiDmStgmb8MAcUzDlRgPHJwf4BkPWPd4kL447gaiMJSw2I0t2LdCr%2BNUCMaRNKqtWH4GPQAqQImGrpimm%2FJgsfLAzEdqTERHefBbFHMNvRncsGOqUBU9NZARDrcipsG3LbiOAWpGr0LQhhxFlw0fPyIemmqQLP8uIyAw7cGdBBPpUIZldoXlN%2F3VE0XtXDxMF3gtPxGIoC%2BuLZMVnhbbfZuRS30CfJDumzFAiGCMITnxlXI09gnb76NoH3fnN1VqclHn7DmmNtnsGV7IGqQdsD7%2BkI8nxyQkC2G68mwW1IRCoKJf%2Bt98RT%2F7ayp%2Bqedf0m2RuCa3du2MJA&X-Amz-Signature=ff3caf913064832eb2a9f703747831cadf5d38682426cf204ec23252bb0d39b5&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject)


其中 ρ 是控制学习率调整程度的常数，在实验中设置为 0.5。$1{\text l}$是指示函数。条件 C1 检查在一段间隔内，干净数据上的损失下降的时期数是否低于阈值。在这种情况下，认为需要降低学习率以促进从干净数据中学习良性特征。条件 C2 检查在训练过程中，干净数据上的最大损失是否保持不变或增加，并且学习率在两个相邻的检查点没有改变。在这种情况下，认为需要改变学习率以防止性能振荡。注意，如果干净数据的准确性增加或保持不变，则学习率不会改变。


算法 1 总结了 SAGE 的整体算法。

