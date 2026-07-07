---
title: "Micrograd - 从零构建自动微分引擎与神经网络"
tags:
  - 深度学习
  - 自动微分
  - 反向传播
  - 神经网络
  - PyTorch
  - Python
  - 微积分
  - 梯度下降
aliases:
  - Karpathy Micrograd
  - micrograd 教程
  - 自动微分引擎
created: 2026-07-07
---

# Micrograd - 从零构建自动微分引擎与神经网络

## 核心概述

本笔记整理自 Andrej Karpathy 的 Micrograd 教程。课程从零开始构建一个标量级别的自动微分引擎（autograd engine），并基于它搭建多层感知机（MLP），最终完成一个完整的神经网络训练流程。

**为什么重要**：反向传播是所有现代深度学习框架的核心算法。理解 Micrograd 可以让你在标量层面直观看懂反向传播的本质，而不被张量运算的工程细节所干扰。

**解决什么问题**：
- 直观理解导数与链式法则在神经网络中的作用
- 理解自动微分引擎的工作原理
- 理解梯度下降训练循环的完整流程
- 理解 PyTorch 等工业级库的底层逻辑

> [!note] 核心论点
> Micrograd 是训练神经网络所需的全部工具，其余一切都只是效率优化。反向传播自动微分引擎仅约 100 行 Python 代码，神经网络库约 150 行代码。数学本质不变，张量只是把标量打包成数组以利用并行计算。

---

## 知识体系

### 1. 导数的直观理解

#### 1.1 导数的本质

导数测量的是函数在某一点的**灵敏度**——当输入微小变化时，输出如何变化。

**数值近似法**（导数定义）：

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

用很小的 $h$（如 0.001）来估算导数：

```python
def f(x):
    return 3 * x**2 - 4 * x + 5

h = 0.001
x = 3.0
# 数值导数
derivative = (f(x + h) - f(x)) / h  # ≈ 14.0
```

- 符号求导（手算 $6x - 4$，代入 $x=3$ 得 14）与数值近似结果一致
- 当 $x = 2/3$ 时导数为零，函数值不响应微小扰动
- 浮点精度有限，$h$ 太小会出现浮点误差

#### 1.2 偏导数

对于多变量函数 $d = a \times b + c$，偏导数表示固定其他变量时，某变量对输出的影响：

| 偏导数 | 解析结果 | 数值验证（a=2, b=3, c=10） |
|--------|---------|--------------------------|
| $\partial d / \partial a$ | $b$ | 3 |
| $\partial d / \partial b$ | $a$ | 2 |
| $\partial d / \partial c$ | $1$ | 1 |

---

### 2. Value 类：构建计算图的数据结构

#### 2.1 基本设计

`Value` 类封装一个标量值，并记录它如何由其他值计算而来：

```python
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

**核心属性**：
- `data`：存储标量值
- `grad`：存储该值对最终输出的导数（初始为 0）
- `_prev`：子节点集合（生成该值的输入值）
- `_op`：产生该值的运算符（如 `'+'`, `'*'`）
- `_backward`：反向传播函数（默认空操作）

#### 2.2 运算符重载

通过 Python 魔术方法实现运算符重载，使 `Value` 对象支持算术运算：

**加法**：

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad
    out._backward = _backward

    return out
```

**乘法**：

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward

    return out
```

> [!important] 自动类型转换
> `other = other if isinstance(other, Value) else Value(other)` 使得 `Value + int` 这样的混合运算成为可能。

#### 2.3 反向反射方法（radd / rmul / rsub 等）

Python 调用 `a * 2` 时会尝试 `a.__mul__(2)`，但 `2 * a` 会先尝试 `int.__mul__(2, a)`，失败后查找 `a.__rmul__(2)`：

```python
def __radd__(self, other):       # other + self
    return self + other

def __rmul__(self, other):       # other * self
    return self * other

def __neg__(self):               # -self
    return self * -1

def __sub__(self, other):        # self - other
    return self + (-other)

def __rsub__(self, other):       # other - self
    return other + (-self)

def __truediv__(self, other):    # self / other
    return self * other**-1

def __rtruediv__(self, other):   # other / self
    return other * self**-1
```

减法、除法都通过已有运算组合实现，减少重复代码。

#### 2.4 幂运算

```python
def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward

    return out
```

局部导数使用**幂法则**：$\frac{d}{dx} x^n = n \cdot x^{n-1}$

> [!note] 限制
> 这里 `other` 必须是 `int` 或 `float`。若指数也是 `Value` 对象，导数表达式会不同（需要用到对数微分法则）。

#### 2.5 激活函数

Micrograd 源码使用 ReLU，课程中演示使用 tanh：

```python
def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out
```

tanh 的实现（课程版本）：

$$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$

其导数为：

$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out
```

> [!tip] 抽象层级的选择
> tanh 可以作为单一函数实现（已知导数 $1-t^2$），也可以分解为指数、加法、乘法、除法等基本运算。两种方式得到相同的前向值和梯度。**在什么层级实现操作完全取决于你**，关键在于你能计算局部导数并应用链式法则。

---

### 3. 反向传播与链式法则

#### 3.1 链式法则

若 $z$ 依赖于 $y$，$y$ 依赖于 $x$，则：

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

**直观类比**：汽车速度是自行车的 2 倍，自行车速度是步行者的 4 倍，则汽车速度是步行者的 $2 \times 4 = 8$ 倍。中间变化率相乘得到整体变化率。

#### 3.2 手动反向传播示例

以 $L = d \times f$，$d = e + c$，$e = a \times b$ 为例（$a=2, b=-3, c=10, f=2$）：

**基础情况**：$dL/dL = 1$

**乘法节点**（$L = d \times f$）：
- $dL/dd = f = 2$
- $dL/df = d = 4$

**加法节点**（$d = e + c$）：
- 局部导数 $dd/dc = 1$，$dd/de = 1$
- 链式法则：$dL/dc = dL/dd \times dd/dc = 2 \times 1 = 2$
- 加法节点的作用是**传递梯度**给所有子节点

**乘法节点**（$e = a \times b$）：
- 局部导数 $de/da = b = -3$，$de/db = a = 2$
- $dL/da = dL/de \times de/da = 2 \times (-3) = -6$
- $dL/db = dL/de \times de/db = 2 \times 2 = 4$

> [!summary] 反向传播本质
> 反向传播 = 沿计算图反向递归应用链式法则。每个节点只需知道：
> 1. 自己的局部导数（该运算的求导规则）
> 2. 上游传来的梯度（输出对当前节点的导数）
> 两者相乘并传递给子节点。

#### 3.3 神经元中的反向传播

单个神经元：$o = \tanh(w_1 x_1 + w_2 x_2 + b)$

反向传播路径：
1. $o$ 的梯度初始化为 1
2. tanh 局部导数：$1 - o^2$，传递给 $n$
3. 加法节点：梯度直接传递给 $x_1w_1$、$x_2w_2$、$b$
4. 乘法节点：$x_i$ 的梯度 = $w_i \times$ 上游梯度；$w_i$ 的梯度 = $x_i \times$ 上游梯度

> [!important] 乘法节点的关键特性
> 当某个输入为 0 时（如 $x_2 = 0$），对应权重的梯度也为 0。因为乘以 0 后输出不随该权重变化——导数反映的是"改变这个变量，输出会怎样变化"。

---

### 4. 自动化反向传播

#### 4.1 _backward 闭包机制

每个运算在创建输出 `Value` 时，同时定义一个 `_backward` 闭包函数，封装该运算的链式法则应用：

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad       # 加法局部导数为1
        other.grad += out.grad
    out._backward = _backward

    return out
```

- 闭包捕获了 `self`、`other`、`out` 的引用
- 叶节点的 `_backward` 默认为空操作（`lambda: None`）

#### 4.2 拓扑排序

反向传播要求：**一个节点的所有后续节点必须先处理完毕**，才能处理该节点。这通过拓扑排序实现。

```python
def backward(self):
    # topological order all of the children in the graph
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # go one variable at a time and apply the chain rule to get its gradient
    self.grad = 1
    for v in reversed(topo):
        v._backward()
```

**拓扑排序保证的不变量**：一个节点只有在所有子节点都进入列表后，自身才会被加入列表。逆序遍历拓扑序列，即从输出端到输入端的正确反向传播顺序。

#### 4.3 梯度累积 bug（关键易错点）

> [!warning] 常见 Bug：梯度覆盖
> 当同一个变量在计算图中被多次使用时（如 `b = a + a`），如果用 `=` 赋值梯度，会导致梯度被覆盖而非累加。

**错误示例**：`b = a + a`，正确梯度应为 2（$1 + 1$），但若用 `=` 则：
- 第一次：`a.grad = 1`
- 第二次：`a.grad = 1`（覆盖，而非累加）
- 结果错误为 1

**修复方案**：所有 `_backward` 中使用 `+=` 而非 `=`：

```python
def _backward():
    self.grad += out.grad       # ✅ 累加
    other.grad += out.grad      # ✅ 累加
```

由于梯度初始化为 0，累加机制天然正确处理了变量复用的情况。

---

### 5. 神经网络库（nn.py）

#### 5.1 Module 基类

对标 PyTorch 的 `nn.Module`：

```python
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
```

#### 5.2 Neuron 类

单个神经元：$o = \text{activation}(w \cdot x + b)$

```python
class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]
```

- `nin`：输入维度
- 权重 $w$ 初始化为 $[-1, 1]$ 均匀分布
- 偏置 $b$ 初始化为 0
- `sum()` 第二参数为起始值，直接从 `self.b` 开始累加

#### 5.3 Layer 类

神经元层是独立神经元的集合，全部连接到相同输入：

```python
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
```

- 若输出仅一个元素，直接返回该值（方便使用）
- 否则返回列表

#### 5.4 MLP 类

多层感知机：将多个层串联：

```python
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

- `nouts` 是各层神经元数量的列表
- 除最后一层外，每层使用非线性激活
- 示例：`MLP(3, [4, 4, 1])` → 3 输入，两层各 4 神经元，1 输出，共 41 个参数

> [!note] API 对标
> Micrograd 的 `Module`、`zero_grad`、`parameters()` 设计完全对标 PyTorch 的 `nn.Module` API。

---

### 6. 损失函数与训练

#### 6.1 均方误差损失（MSE）

```python
# xs: 输入样本, ys: 目标值
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
```

- 每个样本的预测值与真实值相减后平方
- 平方确保结果非负，且偏差越大损失越高
- 当预测值等于目标值时，损失为 0

#### 6.2 梯度下降训练循环

```python
for k in range(steps):
    # 前向传播
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # 反向传播前清零梯度（关键！）
    n.zero_grad()
    loss.backward()

    # 参数更新（沿梯度反方向）
    for p in n.parameters():
        p.data += -learning_rate * p.grad

    print(k, loss.data)
```

> [!warning] 最常见 Bug：忘记清零梯度
> 反向传播中所有 `_backward` 使用 `+=` 累加梯度。如果不清零，梯度会在多次迭代间持续累加，导致训练异常。**每次反向传播前必须调用 `zero_grad()`。**
>
> 这也是 PyTorch 的标准做法：`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`。

#### 6.3 梯度下降的关键概念

| 概念 | 说明 |
|------|------|
| **学习率** | 每步更新的步长。太小收敛慢，太大会越过最优解导致训练不稳定 |
| **梯度方向** | 梯度指向损失**增大**的方向，因此更新时取**负号** |
| **更新公式** | $p_{new} = p_{old} - \eta \cdot \nabla p$ |

> [!tip] 学习率调节
> 学习率调节是一门精细的艺术。实际中常用学习率衰减策略：训练初期用较高学习率快速收敛，后期逐步调低以捕捉细微特征。

---

### 7. PyTorch 对比

#### 7.1 张量 vs 标量

| 特性 | Micrograd | PyTorch |
|------|-----------|---------|
| 数据单元 | 标量 `Value` | N 维张量 `torch.Tensor` |
| 默认精度 | float64 | float32 |
| 梯度需求 | 默认需要 | 默认不需要（`requires_grad=True`） |
| API | 简化版 | 完整工业级 |

```python
import torch

# 创建单元素张量
x1 = torch.Tensor([1.0]).double(); x1.requires_grad = True
# ... 同理创建其他张量

# 前向传播
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

# 反向传播
o.backward()

# 梯度
print(x1.grad)  # tensor([...])
print(x1.grad.item())  # 提取标量值
```

#### 7.2 .item() 方法

PyTorch 的 `.item()` 将单元素张量转换为 Python 标量：

```python
o.item()  # 0.7071...  从 tensor(0.7071) 提取
```

#### 7.3 自定义 autograd 函数

PyTorch 允许通过 `torch.autograd.Function` 添加自定义可微函数：

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 前向计算
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：返回输入的梯度
        x, = ctx.saved_tensors
        return grad_output * local_derivative
```

只要能完成前向计算并掌握局部梯度，就可以与 PyTorch 现有组件无缝集成。

---

### 8. 核心总结

> [!abstract] 神经网络的本质
> 神经网络是数学表达式：
> - **输入**：数据 + 权重参数
> - **前向传播**：数学表达式计算预测值
> - **损失函数**：衡量预测准确性
> - **反向传播**：计算损失对权重的梯度
> - **梯度下降**：沿梯度反方向更新参数，迭代最小化损失

从 41 个参数的小型 MLP 到数百亿参数的 GPT，原理完全相同：
- GPT 的任务是预测序列中的下一个词
- 使用交叉熵损失而非均方误差
- 使用随机梯度下降（SGD）的变体
- 但梯度下降和反向传播的核心逻辑不变

---

## 关键公式速查

| 运算 | 前向 | 局部导数（反向） |
|------|------|-----------------|
| 加法 $a + b$ | $a + b$ | $\partial/\partial a = 1$, $\partial/\partial b = 1$ |
| 乘法 $a \times b$ | $a \times b$ | $\partial/\partial a = b$, $\partial/\partial b = a$ |
| 幂 $a^n$ | $a^n$ | $\partial/\partial a = n \cdot a^{n-1}$ |
| ReLU | $\max(0, x)$ | $\partial/\partial x = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ |
| tanh | $\frac{e^{2x}-1}{e^{2x}+1}$ | $1 - \tanh^2(x)$ |
| exp | $e^x$ | $e^x$ |

---

## 易错点汇总

1. **梯度累积 vs 覆盖**：`_backward` 中必须用 `+=`，否则变量复用时梯度会被覆盖
2. **忘记清零梯度**：每次 `backward()` 前必须 `zero_grad()`，否则梯度跨迭代累加
3. **更新方向取反**：梯度指向损失增大方向，更新时必须取负号
4. **学习率过大**：可能越过最优解，导致训练不稳定甚至发散
5. **`__rmul__` 缺失**：`2 * a` 需要 `__rmul__`，否则报错（`int` 没有 `Value` 运算能力）
6. **幂运算指数类型**：`__pow__` 中 `other` 必须是 `int/float`，不能是 `Value`
7. **浮点精度**：数值梯度检查时 $h$ 太小会出现浮点误差

---

## 相关链接

- **Micrograd GitHub**: [karpathy/micrograd](https://github.com/karpathy/micrograd)
- **PyTorch 文档**: [torch.autograd](https://pytorch.org/docs/stable/autograd.html)
- **视频原址**: [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0)

---

## 参考来源

- [吴恩达深度学习手写笔记](http://www.kerwin.cn/dl/detail/zj93170/2047961)
- [训练神经网络快速指南 - 知乎](https://zhuanlan.zhihu.com/p/540117244)
- [AI大神吴恩达老师 DeepLearning 系列课程学习笔记 - 知乎](https://zhuanlan.zhihu.com/p/553471322)
