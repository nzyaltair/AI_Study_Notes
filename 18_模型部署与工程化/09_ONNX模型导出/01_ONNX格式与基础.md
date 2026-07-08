# ONNX 格式与基础

## 背景与发展

深度学习框架百花齐放，PyTorch、TensorFlow、MXNet 等各有优势，但模型格式不互通导致部署困难：用 PyTorch 训练的模型无法直接在 TensorFlow 生态中运行，反之亦然。2017 年，Facebook（Meta）与 Microsoft 联合发起 [[ONNX]]（Open Neural Network Exchange）项目，旨在定义一套开放的模型表示标准，使模型能在不同框架和硬件平台间无缝迁移。项目现由 Linux 基金会维护，已成为工业界事实上的模型交换格式。

## 核心思想

ONNX 的核心思想是定义一套统一的中间表示（IR）：将深度学习模型抽象为**有向无环计算图**（Computational Graph），图中节点表示算子（Operator），边表示张量（Tensor）数据流。无论模型用哪个框架训练，只要能导出为符合 ONNX IR 的计算图，就能被任何支持 ONNX 的推理引擎执行。

- **框架无关**：PyTorch、TensorFlow、PaddlePaddle 等均可导出
- **开放标准**：由社区共同维护，不受单一厂商控制
- **跨平台**：支持 Windows、Linux、macOS 及移动设备
- **优化推理**：[[ONNX Runtime]] 提供高性能推理引擎，自动执行图优化和算子融合

## 技术原理

### Opset 版本

ONNX 通过 **Operator Set（Opset）** 版本管理算子定义。每个 Opset 版本定义了一组算子及其语义，版本号递增意味着新增算子或功能扩展。

| Opset | 发布时间 | 关键特性 | 推荐场景 |
|-------|---------|---------|---------|
| 10 | 2019-12 | 基础算子 | 老系统兼容 |
| 11 | 2020-06 | Bool 类型、动态形状基础 | 通用基线 |
| 13 | 2020-12 | 量化算子、改进动态形状 | 通用推荐 |
| 14 | 2021-03 | 目标检测优化 | 检测模型 |
| 15 | 2021-06 | 新数学函数（Cos/Sin） | 最新功能 |

选择策略：目标部署环境已知时选择其支持的最高 Opset；多平台部署选 Opset 11-13 作为基线。

### IR 与序列化

ONNX 模型文件使用 **Protocol Buffers（protobuf）** 作为序列化格式。模型结构包含：

- **ModelProto**：顶层容器，包含 ir_version、opset_import、metadata
- **GraphProto**：计算图定义，包含 node、input、output、initializer
- **NodeProto**：单个算子节点，包含 op_type、input、output、attributes
- **TensorProto**：权重和偏置等初始器数据

### 最小导出示例

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel().eval()
dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
```

### ONNX Runtime 推理

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
input_data = np.random.randn(1, 10).astype(np.float32)
outputs = session.run(None, {"input": input_data})
```

### 环境搭建

核心依赖：`onnx`（模型格式库）、`onnxruntime`（推理引擎）、训练框架（`torch` 或 `tensorflow`）。GPU 环境需匹配 CUDA/cuDNN 版本，安装 `onnxruntime-gpu`。推荐使用 conda 或 venv 隔离环境，避免依赖冲突。

## 发展演进

- 2017：ONNX 1.0 发布，支持基础 CNN 模型
- 2019-2020：Opset 10-13，引入动态形状、量化算子
- 2021-2022：Opset 14-16，增强控制流和符号形状推导
- 2023+：Opset 17+，持续扩展算子覆盖；ONNX Runtime 成为多硬件后端统一接口

## 关键算法·模型

ONNX 本身是格式标准而非算法，但其生态中的核心工具链值得关注：

- **onnx-simplifier**：通过常量折叠和形状推断简化计算图，减少冗余节点
- **onnx-graphsurgeon**：NVIDIA 开发的图编辑库，支持节点增删改、子图替换
- **onnxruntime-extensions**：扩展 ONNX Runtime 支持非标准算子（如 GridSample、Unique）

## 应用场景

- **训练-部署分离**：研究团队用 PyTorch 训练，部署团队用 ONNX Runtime 推理
- **硬件适配**：一次转换，多平台运行（CPU/GPU/NPU）
- **版本管理**：标准化格式简化模型版本控制
- **性能优化**：ONNX Runtime 自动执行图优化（算子融合、常量折叠、内存复用）

## 与其他技术关系

- [[模型部署]] — ONNX 是部署流水线中的模型交换环节
- [[模型推理]] — ONNX Runtime 是主流推理引擎之一
- [[TensorRT]] — NVIDIA GPU 上的极致优化后端，可直接加载 ONNX 模型
- [[量化]] — ONNX Opset 13+ 包含量化算子定义，支持 INT8/FP16 推理
- [[OpenVINO]] — Intel 硬件上的推理引擎，支持 ONNX 输入
