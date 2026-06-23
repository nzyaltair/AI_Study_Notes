---
aliases: [microgpt, 微型GPT, Andrej Karpathy, GPT从零开始]
created: 2026-02-12
tags: [deep-learning, transformers, nlp, gpt, python, from-scratch, autograd, attention]
category: tutorial
---

# microgpt：200行Python手写GPT

> 仅200行纯Python实现GPT训练与推理，无任何依赖。理解[[LLM]]算法本质的最佳起点。

[[Karpathy]]出品

## 资源
- 源码：https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
- 教程页：https://karpathy.ai/microgpt.html
- Colab：https://colab.research.google.com/drive/1vyN5zo6rqUp_dYNbT4Yrco66zuWCZKoN

## 快速开始

```bash
python train.py
# 约1分钟完成训练，loss从3.3降至2.37，生成32个新名字
```

## 完整源代码

```python
"""
以纯Python（无依赖）编写的最精简的GPT训练与推理实现。
这是完整的算法实现。其余的一切都只是效率优化。
"""
import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42)  

# 创建数据集 `docs`: 文档的列表[str]
if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt',
        'input.txt'
    )
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)

# Tokenizer：字符 → 整数ID
uchars = sorted(set(''.join(docs)))  # 唯一字符 a-z
BOS = len(uchars)                    # 特殊token：序列开始
vocab_size = len(uchars) + 1         # 26 + 1 = 27

# 自动微分：Value类定义
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data; self.grad = 0
        self._children = children; self._local_grads = local_grads
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def backward(self):
        topo = []; visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children: build_topo(child)
                topo.append(v)
        build_topo(self); self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

# 模型参数初始化
n_layer = 1; n_embd = 16; block_size = 16; n_head = 4
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]

# 辅助函数
def linear(x, w): return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    return [e / sum(exps) for e in exps]
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# 模型前向传播（单token位置）
def gpt(token_id, pos_id, keys, values):
    x = [t + p for t, p in zip(state_dict['wte'][token_id], state_dict['wpe'][pos_id])]
    x = rmsnorm(x)
    for li in range(n_layer):
        # 多头注意力
        x_residual = x; x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k); values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_scores = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_scores)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # MLP
        x_residual = x; x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

# Adam优化器
learning_rate, beta1, beta2, eps = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params); v = [0.0] * len(params)

# 训练循环
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())
    loss = sum(losses) / len(losses)
    loss.backward()
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
        p.grad = 0
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# 推理生成
temperature = 0.5
print("\n--- 生成新名字 ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS; sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS: break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

## 核心概念

- 数据：32,000个人名（每行一个）
- 字符级tokenizer：26个字母 + 1个BOS token
- Tokenization：`"emma"` → `[BOS, 'e','m','m','a', BOS]`

### 2. 自动微分引擎

`Value`类包装标量并构建[[计算图]]：

```python
class Value:
    def __add__(self, other): ...
    def __mul__(self, other): ...  # 记住局部梯度
    def backward(self): ...         # 拓扑排序后链式传播
```

每个运算记录局部导数，`backward()`按反向拓扑顺序应用链式法则：
`child.grad += local_grad * parent.grad`

### 3. [[Transformer]]架构

`gpt(token_id, pos_id, keys, values)` 为单个token位置的完整前向传播：

```
嵌入 → [N层 × (注意力 + MLP)] → 输出logits
```

- **嵌入**：token + position 向量相加
- **多头注意力**：Q,K,V投影 → 缓存到KV Cache → scaled dot-product → 投影 + 残差
- **MLP**：线性 → ReLU → 线性 + 残差
- **RMSNorm**：每层前归一化（替代LayerNorm）
- **KV Cache**：显式累积历史K,V（训练时仍在计算图内）

### 4. 训练

```python
loss = cross_entropy(softmax(logits), target)
loss.backward()
Adam.update(params)   # 含梯度清零
```

- 目标：最大化下一个字符的似然
- 优化：Adam（动量+自适应lr，含偏差校正）
- 学习率：线性衰减 0.01 → 0

### 5. 推理

从BOS开始，迭代采样直到BOS或达到max_len：

```python
logits = gpt(prev_token, pos, keys, values)
probs = softmax(logits / temperature)
next_token = random.choices(probs)
```

`temperature` 控制随机性：越小越确定，越大越随机。

## 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_layer` | 1 | 层数 |
| `n_embd` | 16 | 嵌入维度 |
| `n_head` | 4 | 注意力头数 |
| `block_size` | 16 | 序列长度 |
| `num_steps` | 1000 | 训练步数 |

总参数量：**4,192**

## 项目构建历程

| 版本 | 内容 |
|------|------|
| `train0.py` | 双字母组计数表 |
| `train1.py` | MLP + 手动梯度 + SGD |
| `train2.py` | 自动微分（Value类） |
| `train3.py` | 位置嵌入 + 单头注意力 + RMSNorm |
| `train4.py` | 多头注意力 + 层循环（完整GPT） |
| `train5.py` | Adam优化器（即本文件）|

[build_microgpt.py](https://gist.github.com/karpathy/561ac2de12a47cc06a23691e1be9543a) 包含所有版本差异。

## 生产级对比

| 方面 | microgpt | 生产LLM（GPT-4） |
|------|----------|------------------|
| 数据 | 32K名字 | 万亿token网络文本 |
| 分词 | 字符级（27词表） | BPE子词（~100K） |
| 架构 | 1层，4K参数 | 100+层，数千亿参数 |
| 训练 | 单GPU，1分钟 | 数千GPU，数月 |
| 推理 | 纯Python，慢 | 优化CUDA内核，高速 |

**本质相同**：都是next-token prediction，都有注意力+MLP+残差结构。

## FAQ

**Q: 模型真的"理解"吗？**  
A: 没有魔法。只是映射token到概率分布的数学函数。参数调整使正确next token概率最大。是否算"理解"是哲学问题，但机制完全在这200行中。

**Q: 为什么有效？**  
A: 数千参数在大量步骤中被优化，捕捉数据的统计规律（如"qu"常共现、名字多从辅音开始等）。模型学的是概率分布，不是显式规则。

**Q: 与ChatGPT的关系？**  
A: 完全相同核心循环（预测下一个token、采样、重复）的规模化扩展，加上SFT和RL后训练。

**Q: "幻觉"是什么？**  
A: 模型从概率分布采样，没有真实概念。生成"karia"这样的名字，与ChatGPT编造错误事实是同种现象：统计上合理但实际不存在。

**Q: 为什么这么慢？**  
A: 纯Python逐标量处理。GPU可并行百万标量，快数个数量级。

**Q: 如何提升质量？**  
A: 增加训练步数、模型规模（`n_embd`, `n_layer`）、数据集大小。

**Q: 换数据集会怎样？**  
A: 模型会学任何数据分布。只需将`input.txt`换成城市名、Pokemon名、英文词等，其余代码不变。

## 参考

- microgpt源码：https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
- micrograd（自动微分教程）：https://github.com/karpathy/micrograd
- makemore（字符级语言模型项目）：https://github.com/karpathy/makemore
- The Illustrated Transformer：http://jalammar.github.io/ illustrated-transformer/
- Attention Is All You Need (原始论文)

---

*最后更新：2026-02-12*
