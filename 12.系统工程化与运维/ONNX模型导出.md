# ONNX模型导出

## 背景

ONNX（Open Neural Network Exchange）是微软和Facebook联合推出的开放模型格式，旨在实现不同[[深度学习框架|深度学习框架]]间的模型互操作性。通过ONNX，可以用PyTorch训练模型，然后部署到TensorRT、OpenVINO等推理引擎，实现跨平台高性能推理。

## 核心思想

将不同框架的模型统一为标准中间表示（IR），通过ONNX Runtime或下游推理引擎执行，解耦模型训练与部署。

## 技术原理

### ONNX导出流程

```
PyTorch/TensorFlow模型 → ONNX导出 → ONNX模型 → ONNX Runtime/TensorRT/OpenVINO → 推理
```

### ONNX核心概念

| 概念 | 说明 |
|------|------|
| **IR（中间表示）** | 标准化的计算图格式 |
| **OpSet** | 算子版本集合 |
| **动态维度** | 支持可变batch/序列长度 |
| **ONNX Runtime** | 跨平台推理引擎 |

### 导出示例

```python
import torch

# PyTorch模型导出为ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,                          # PyTorch模型
    dummy_input,                    # 示例输入
    "model.onnx",                   # 输出路径
    export_params=True,             # 导出参数
    opset_version=17,               # OpSet版本
    do_constant_folding=True,       # 常量折叠优化
    input_names=['input'],          # 输入名
    output_names=['output'],        # 输出名
    dynamic_axes={                  # 动态维度
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### 导出挑战与解决

| 挑战 | 原因 | 解决方案 |
|------|------|---------|
| **不支持算子** | 自定义算子未在ONNX注册 | 注册自定义算子/替换实现 |
| **动态形状** | 可变输入尺寸 | 使用dynamic_axes |
| **控制流** | if/for循环 | ONNX Loop/If算子 |
| **精度差异** | FP32→FP16精度损失 | 量化校准 |
| **流式模型** | KV-Cache等动态状态 | 分步导出 |

### ONNX优化

- **图优化**：算子融合、常量折叠、冗余消除
- **量化**：动态量化（INT8）/静态量化（INT8/UINT8）
- **执行提供者**：CUDA、TensorRT、OpenVINO后端加速
- **性能对比**：ONNX Runtime vs 原生框架推理速度

### 验证与评估

```python
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
session = ort.InferenceSession("model.onnx")

# 推理
inputs = {session.get_inputs()[0].name: np.random.randn(1, 3, 224, 224).astype(np.float32)}
outputs = session.run(None, inputs)

# 与原始PyTorch模型对比
torch_output = model(torch.from_numpy(inputs['input'])).detach().numpy()
assert np.allclose(outputs[0], torch_output, atol=1e-5)
```

## 详细教程

ONNX导出的完整教程（基础概念、流式/非流式转换、跨框架转换、常见问题、验证评估）详见：

- [[00-导航索引]] - ONNX导出教程导航
- [[01-基础概念与环境准备]] - 基础概念与环境准备
- [[02-非流式模型转换]] - 非流式模型转换
- [[03-流式模型转换]] - 流式模型转换
- [[04-跨框架转换]] - 跨框架转换
- [[05-常见问题解决]] - 常见问题解决
- [[06-验证与评估]] - 验证与评估
- [[07-场景优化建议]] - 场景优化建议

## 发展演进

ONNX 1.0（2017）→ OpSet版本迭代 → ONNX Runtime → 量化支持 → 动态形状 → 流式LLM导出

## 应用领域

- **跨框架部署**：PyTorch训练 → TensorRT部署
- **边缘推理**：ONNX Runtime Mobile
- **模型量化**：INT8量化加速推理
- **[[模型部署与服务化|LLM部署]]**：LLM ONNX导出用于推理优化

## 与其他技术关系

- ONNX解耦了[[深度学习框架|深度学习框架]]和推理引擎
- [[LLM推理与优化|LLM推理优化]]可借助ONNX量化
- [[容器化与Docker|容器化部署]]中ONNX模型作为制品
- [[模型部署与服务化|模型服务化]]可使用ONNX Runtime后端
