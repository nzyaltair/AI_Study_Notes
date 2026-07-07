# Python基础

## 背景

Python凭借简洁语法、丰富库生态和强大可读性，成为AI开发的首选语言。AI主流框架（PyTorch、TensorFlow）和数据处理工具（NumPy、Pandas）均以Python为主要接口。

## 核心思想

Python是高级解释型语言，其动态类型、垃圾回收和丰富的标准库使其特别适合快速原型开发和数据处理。在AI开发中，Python作为"胶水语言"连接底层C/C++高性能计算和上层算法逻辑。

## 技术原理

### 数据类型与结构

- **基本类型**：int、float、str、bool
- **容器类型**：
  - list：有序可变序列，`[1, 2, 3]`
  - tuple：有序不可变序列，`(1, 2, 3)`
  - dict：键值对映射，`{"key": "value"}`
  - set：无序不重复集合，`{1, 2, 3}`
- **推导式**：`[x**2 for x in range(10) if x % 2 == 0]`

### 函数与装饰器

- **函数**：`def func(args): return result`
- **Lambda表达式**：`lambda x: x**2`
- **装饰器**：在不修改函数的前提下扩展功能
  ```python
  def timer(func):
      def wrapper(*args, **kwargs):
          start = time.time()
          result = func(*args, **kwargs)
          print(f"耗时: {time.time() - start:.4f}s")
          return result
      return wrapper
  ```
- **生成器**：使用`yield`实现惰性求值，节省内存

### 面向对象编程

- **类与对象**：封装数据和行为
- **继承**：复用父类代码
- **多态**：同一接口不同实现
- **魔术方法**：`__init__`、`__call__`、`__getitem__`等

### 异常处理

```python
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"值错误: {e}")
except Exception as e:
    logger.error(f"未知错误: {e}")
finally:
    cleanup()
```

### 模块与包

- **模块**：一个`.py`文件
- **包**：包含`__init__.py`的目录
- **导入**：`import module`、`from module import func`、`from module import *`

### 类型注解

```python
from typing import List, Dict, Optional, Tuple

def process_data(
    features: List[float],
    labels: Optional[List[int]] = None,
    config: Dict[str, any] = {}
) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

### 并发编程

- **多线程**：`threading`模块，受GIL限制，适合I/O密集任务
- **多进程**：`multiprocessing`模块，绕过GIL，适合CPU密集任务
- **异步IO**：`asyncio`模块，协程实现高并发
- **并行计算**：`concurrent.futures`提供统一的并行接口

## 应用领域

- **[[深度学习]]**：PyTorch/TensorFlow模型开发
- **[[机器学习]]**：Scikit-learn数据处理与模型训练
- **[[数据处理与工程化|数据工程]]**：Pandas/NumPy数据处理
- **[[Agent智能体|AI应用]]**：LLM API调用、Agent开发

## 与其他技术关系

- Python是[[Python用于机器学习|ML库]]的使用接口
- Python与[[Linux与开发环境|Linux]]配合是AI开发的标准环境
- Python的NumPy底层依赖[[线性代数]]运算
