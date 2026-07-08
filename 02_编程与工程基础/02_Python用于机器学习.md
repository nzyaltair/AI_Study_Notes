# Python 用于机器学习

## 背景与发展

Python 机器学习生态由 NumPy（数值计算）、Pandas（结构化数据）、Scikit-learn（经典 ML）、PyTorch/TensorFlow（深度学习）构成完整工具链。这些库通过统一的 Python 接口协同工作，覆盖从数据准备到模型部署的全流程。

## 核心思想

Python ML 生态采用分层架构：NumPy 提供底层 n 维数组计算，Pandas 提供表格数据抽象，Scikit-learn 提供经典 ML 算法的统一 API，PyTorch/TensorFlow 提供自动微分和 GPU 加速的深度学习框架。各层之间通过 ndarray/Tensor 数据格式自然衔接。

## 技术原理

### NumPy 核心机制

- **ndarray**：同类型元素的 n 维数组，连续内存布局，支持向量化运算
- **广播机制**：不同形状数组间的运算规则，自动扩展低维数组
- **视图与拷贝**：切片产生视图（共享内存）， Fancy Indexing 产生拷贝
- **关键操作**：`np.dot`（矩阵乘法）、`np.linalg.svd`（SVD 分解）、`np.einsum`（张量缩并）

### Scikit-learn 统一 API

所有估计器遵循 `fit()` / `predict()` / `transform()` 接口，Pipeline 将预处理和模型串联为可复现流程。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])
pipe.fit(X_train, y_train)
```

模型选择通过 `cross_val_score`（交叉验证）和 `GridSearchCV`（超参数搜索）完成。

### PyTorch 核心机制

- **Tensor**：类似 ndarray，支持 GPU 加速和自动微分
- **autograd**：动态计算图引擎，前向传播记录操作，`backward()` 自动求导
- **nn.Module**：神经网络模块基类，通过 `__call__` 调用 `forward()`，自动管理参数注册

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_h)
        self.fc2 = nn.Linear(dim_h, dim_out)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
```

### 训练循环

标准训练流程：前向传播 → 计算损失 → 反向传播 → 参数更新。

```python
for X, y in dataloader:
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```

### 数据加载

`Dataset` 定义数据访问协议（`__getitem__`、`__len__`），`DataLoader` 提供 batch 采样、多进程加载和内存预取（`pin_memory`）。

### GPU 加速

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

数据与模型需位于同一设备，跨设备操作会触发隐式拷贝。

## 发展演进

NumPy（2005）→ Scikit-learn（2007）→ Pandas（2008）→ Theano（2007）→ TensorFlow（2015）→ PyTorch（2016）。PyTorch 凭借动态计算图和直觉式 API 成为学术界主流，2.0 引入 `torch.compile` 以编译优化弥补静态图性能优势。

## 关键算法·模型

- **NumPy**：广播规则、向量化（ufunc）、einsum 求和约定
- **PyTorch autograd**：反向模式自动微分、计算图构建与梯度累积
- **Scikit-learn**：Pipeline、ColumnTransformer、交叉验证与超参数搜索

## 应用场景

- 经典 ML：Scikit-learn 实现 SVM、随机森林、梯度提升等算法
- 深度学习：PyTorch/TensorFlow 模型定义与训练
- 大语言模型：Hugging Face Transformers 加载预训练模型与微调
- 数据工程：Pandas/NumPy 数据清洗、变换、特征构建

## 与其他技术关系

- NumPy 底层运算依赖 [[线性代数]]，SVD/QR 分解调用 LAPACK
- PyTorch 的 autograd 基于链式法则，是 [[深度学习]] 反向传播的实现
- Scikit-learn 的算法实现基于 [[机器学习]] 统计学习理论
- 优化器（SGD、Adam）基于 [[优化方法]] 梯度下降理论
