# Python 基础

## 背景与发展

Python 凭借简洁语法、丰富库生态和"胶水语言"特性，成为 AI 开发的事实标准语言。主流深度学习框架（PyTorch、TensorFlow）和数据处理工具（NumPy、Pandas）均以 Python 为主要接口，底层计算由 C/C++ 和 CUDA 实现，Python 负责上层算法逻辑与实验编排。

## 核心思想

Python 作为 AI 工程语言的核心在于：以解释型动态语言提供快速原型开发能力，通过 C 扩展机制桥接高性能数值计算，以统一接口串联数据处理、模型训练、推理部署全流程。AI 开发中关注的关键语言特性包括向量化运算、自动微分、类型注解与数据类。

## 技术原理

### NumPy 张量操作

NumPy 的 ndarray 是所有 Python ML 库的基础数据结构，底层为连续内存的 C 数组，支持向量化运算与广播机制。

```python
import numpy as np
# 广播：(3,) + (2,3) → (2,3)
a = np.array([1, 2, 3])
b = np.array([[10], [20]])
c = a + b  # [[11,12,13],[21,22,23]]
```

向量化运算避免 Python 循环，性能提升可达 100 倍。矩阵乘法 `np.dot`、SVD 分解 `np.linalg.svd` 等操作直接调用 BLAS/LAPACK。

### PyTorch 张量与自动微分

PyTorch Tensor 在 ndarray 基础上增加 GPU 加速和自动微分（autograd）。

```python
import torch
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()       # 反向传播，自动计算梯度
print(x.grad)       # 2x，基于链式法则
```

autograd 构建动态计算图，前向传播时记录操作，反向传播时自动求导。`requires_grad=True` 标记需要梯度的张量，`backward()` 触发梯度计算。

### 类型注解

类型注解提升代码可读性和 IDE 支持能力，在大型 AI 项目中尤为重要。

```python
from typing import Optional
import numpy as np

def process(
    features: list[float],
    labels: Optional[list[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

mypy 可在编译期进行静态类型检查，catching 类型错误。

### dataclass

dataclass 自动生成 `__init__`、`__repr__` 等方法，适合管理模型配置和超参数。

```python
from dataclasses import dataclass

@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
```

### 魔术方法与协议

Python 的鸭子类型通过魔术方法实现运算符重载和协议，PyTorch 的 `nn.Module` 依赖 `__call__` 实现前向传播，Dataset 依赖 `__getitem__` 和 `__len__` 实现数据加载。

### 并发模型

- **多线程**（`threading`）：受 GIL 限制，适合 I/O 密集任务（数据加载、API 调用）
- **多进程**（`multiprocessing`）：绕过 GIL，适合 CPU 密集任务（数据预处理）
- **异步 IO**（`asyncio`）：协程实现高并发，适合 LLM API 批量调用

## 发展演进

Python 3.5 引入类型注解（PEP 484），3.7 引入 dataclass（PEP 557），3.10 引入结构化模式匹配。PyTorch 1.0（2018）定义了动态图深度学习框架的标准范式，2.0（2023）引入 `torch.compile` 进一步融合编译优化。

## 关键算法·模型

- **NumPy**： Broadcasting 规则、ufunc 向量化、einsum 爱因斯坦求和约定
- **PyTorch autograd**：反向模式自动微分、计算图构建与释放
- **torch.compile**：基于 TorchDynamo 的图捕获与 kernel fusion

## 应用场景

- 深度学习模型开发：PyTorch/TensorFlow 模型定义与训练
- 数据处理流水线：Pandas/NumPy 数据清洗与变换
- LLM 应用：Hugging Face Transformers、API 调用与 Agent 编排
- 实验管理：脚本化超参数搜索与训练控制

## 与其他技术关系

- Python 是 [[机器学习]] 库的统一接口层，底层依赖 C/C++/CUDA 实现
- NumPy 的矩阵运算依赖 [[线性代数]] 理论
- PyTorch 的 autograd 基于微积分中的链式法则
- 类型注解与 dataclass 支撑大型 AI 项目的工程化实践
