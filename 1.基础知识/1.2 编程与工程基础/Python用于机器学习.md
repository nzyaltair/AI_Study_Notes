# Python用于机器学习

## 背景

Python的机器学习生态系统包括NumPy（数值计算）、Pandas（数据处理）、Scikit-learn（经典ML）、PyTorch/TensorFlow（深度学习）等核心库，构成了从数据准备到模型部署的完整工具链。

## 核心思想

Python ML生态的核心是：NumPy提供高效的n维数组计算，Pandas提供结构化数据处理，Scikit-learn提供经典ML算法实现，PyTorch/TensorFlow提供深度学习框架。这些工具通过统一的Python接口协同工作。

## 技术原理

### NumPy核心

- **ndarray**：n维数组，同类型元素，支持向量化运算
- **广播机制**：不同形状数组间的运算规则
- **关键操作**：`np.dot`（矩阵乘法）、`np.linalg.svd`（SVD分解）、`np.random`（随机采样）
- **性能**：底层C实现，比纯Python快100倍

### Pandas核心

- **DataFrame**：二维表格数据结构，类似SQL表
- **关键操作**：
  - 数据读取：`pd.read_csv()`、`pd.read_json()`
  - 数据筛选：`df[df['col'] > 0]`
  - 分组聚合：`df.groupby('col').agg({'val': 'mean'})`
  - 缺失值处理：`df.fillna()`、`df.dropna()`

### Scikit-learn核心

- **统一API**：`fit()` → `predict()` / `transform()`
- **Pipeline**：将预处理和模型串联
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
- **模型选择**：`cross_val_score`、`GridSearchCV`

### PyTorch核心

- **Tensor**：类似NumPy ndarray，支持GPU加速和自动微分
- **autograd**：自动微分引擎
  ```python
  x = torch.randn(3, requires_grad=True)
  y = (x ** 2).sum()
  y.backward()  # 自动计算梯度
  print(x.grad)  # 2x
  ```
- **nn.Module**：神经网络模块基类
- **训练循环**：前向传播 → 计算损失 → 反向传播 → 参数更新

### 数据加载与预处理

```python
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### GPU加速

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

## 发展演进

NumPy（2005）→ Scikit-learn（2007）→ Pandas（2008）→ Theano（2007）→ TensorFlow（2015）→ PyTorch（2016）。PyTorch凭借动态计算图和直观API成为学术界主流，TensorFlow在生产部署领域保持优势。

## 应用领域

- **[[机器学习]]**：Scikit-learn实现经典ML算法
- **[[深度学习]]**：PyTorch/TensorFlow模型开发
- **[[数据处理与工程化|数据工程]]**：Pandas/NumPy数据处理
- **[[大语言模型]]**：Hugging Face Transformers库

## 与其他技术关系

- NumPy底层实现依赖[[线性代数]]运算
- PyTorch的autograd依赖[[微积分|链式法则]]
- Scikit-learn的算法实现基于[[统计学习方法]]
- 深度学习框架的优化器基于[[优化方法]]
