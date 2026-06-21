# Python基础

## 1. 引言：Python在AI开发中的地位

Python是一种高级、解释型、通用的编程语言，凭借其简洁的语法、丰富的库生态和强大的可读性，已成为人工智能（AI）开发的首选语言。在AI领域，Python具有以下显著优势：

- **丰富的AI库生态**：拥有TensorFlow、PyTorch、Scikit-learn等主流AI框架
- **简洁易读的语法**：降低开发门槛，提高开发效率
- **强大的数据处理能力**：NumPy、Pandas等库提供高效的数据处理支持
- **跨平台兼容性**：可在Windows、Linux、macOS等多种平台运行
- **活跃的社区支持**：持续更新和改进，解决问题资源丰富

本文件将系统介绍Python基础知识，结合AI开发场景，为AI开发者提供全面的Python编程参考。

## 2. Python语言基础

### 2.1 安装与环境配置

```bash
# 下载Python（推荐使用3.8+版本）
# 访问 https://www.python.org/downloads/ 下载对应平台的安装包

# 验证安装
python --version
python3 --version

# 安装pip（Python包管理工具）
python -m ensurepip --upgrade

# 验证pip安装
pip --version
pip3 --version
```

### 2.2 基本语法规则

#### 注释

```python
# 单行注释

"""
多行注释
用于详细说明函数、类或模块
"""

# AI开发中注释的最佳实践：
# 1. 对复杂算法和业务逻辑添加注释
# 2. 注释应解释"为什么"，而不仅仅是"是什么"
# 3. 使用文档字符串为函数和类提供详细说明
```

#### 缩进与代码块

```python
# Python使用缩进来表示代码块（4个空格是标准约定）
if True:
    print("缩进正确")
    if True:
        print("嵌套缩进")
else:
    print("缩进错误会导致语法错误")
```

#### 变量与赋值

```python
# 变量赋值
x = 10
name = "Python"
is_valid = True

# 多变量赋值
a, b, c = 1, 2, 3

# 链式赋值
d = e = f = 5

# AI开发中变量命名建议：
# 1. 使用有意义的变量名
# 2. 遵循小写字母加下划线的命名规范
# 3. 避免使用单个字符（除非是循环变量）
learning_rate = 0.001  # 学习率
batch_size = 32        # 批次大小
epoch = 100            # 训练轮数
```

### 2.3 数据类型

#### 基本数据类型

```python
# 整数
age = 25

# 浮点数
temperature = 36.5
pi = 3.1415926

# 字符串
name = "Python"
welcome = 'Welcome to AI development'

# 布尔值
is_active = True
is_empty = False

# None类型（表示空值）
result = None
```

#### 复合数据类型

##### 列表（List）
```python
# 列表创建
numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "cherry"]
mixed = [1, "apple", True, 3.14]

# 列表操作
numbers.append(6)        # 添加元素到末尾
numbers.insert(0, 0)     # 在指定位置插入元素
numbers.remove(3)        # 删除指定元素
numbers.pop()            # 删除并返回最后一个元素
numbers.sort()           # 排序

# 列表切片
first_three = numbers[:3]  # [0, 1, 2]
last_two = numbers[-2:]    # [4, 5]
every_other = numbers[::2]  # [0, 2, 4]

# AI开发中的列表应用
# 存储训练数据标签
y_train = [0, 1, 0, 1, 1]
# 存储模型参数
weights = [0.5, 0.3, 0.2]
```

##### 元组（Tuple）
```python
# 元组创建（不可变）
point = (10, 20)
colors = ("red", "green", "blue")

# 单元素元组需要加逗号
single = (5,)

# 元组解包
x, y = point

# AI开发中的元组应用
# 存储模型的输入输出形状
input_shape = (28, 28, 3)
output_shape = (10,)
```

##### 字典（Dictionary）
```python
# 字典创建
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# 字典操作
person["email"] = "alice@example.com"  # 添加键值对
person["age"] = 31                     # 修改值
del person["city"]                     # 删除键值对
email = person.get("email", "default@example.com")  # 安全获取值

# 遍历字典
for key, value in person.items():
    print(f"{key}: {value}")

# AI开发中的字典应用
# 存储模型超参数
hyperparams = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam"
}
```

##### 集合（Set）
```python
# 集合创建（无序、唯一元素）
numbers = {1, 2, 3, 4, 5}
duplicates = {1, 2, 2, 3, 3, 3}
print(duplicates)  # {1, 2, 3}

# 集合操作
numbers.add(6)           # 添加元素
numbers.remove(3)        # 删除元素
numbers.update({7, 8, 9})  # 添加多个元素

# 集合运算
a = {1, 2, 3}
b = {3, 4, 5}
print(a | b)  # 并集 {1, 2, 3, 4, 5}
print(a & b)  # 交集 {3}
print(a - b)  # 差集 {1, 2}

# AI开发中的集合应用
# 去除重复的类别标签
unique_labels = set(y_train)
```

### 2.4 字符串操作

```python
# 字符串拼接
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name  # "John Doe"

# 字符串格式化
# 1. f-string（Python 3.6+，推荐）
age = 25
message = f"My name is {full_name} and I'm {age} years old"

# 2. format方法
message = "My name is {} and I'm {} years old".format(full_name, age)

# 3. 占位符（旧式，不推荐）
message = "My name is %s and I'm %d years old" % (full_name, age)

# 字符串方法
text = "  Python for AI Development  "
print(text.strip())          # "Python for AI Development"
print(text.lower())          # "  python for ai development  "
print(text.upper())          # "  PYTHON FOR AI DEVELOPMENT  "
print(text.replace("AI", "Artificial Intelligence"))

# 字符串分割与连接
words = text.split()         # ["Python", "for", "AI", "Development"]
joined = "-".join(words)    # "Python-for-AI-Development"

# AI开发中的字符串操作
# 处理文本数据
texts = ["I love AI", "Python is great", "Machine learning is fun"]
processed_texts = [text.lower().strip() for text in texts]
```

## 3. 控制流

### 3.1 条件语句

```python
# 基本if-elif-else结构
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Your grade is: {grade}")

# 嵌套条件
if score >= 60:
    print("Pass")
    if score >= 90:
        print("Excellent!")
else:
    print("Fail")

# AI开发中的条件判断
# 检查模型训练是否需要早停
early_stop = False
if current_loss > previous_loss and patience_counter >= patience:
    early_stop = True
    print("Early stopping triggered")
```

### 3.2 循环语句

#### for循环

```python
# 遍历列表
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# 遍历字典
person = {"name": "Alice", "age": 30}
for key, value in person.items():
    print(f"{key}: {value}")

# 使用range()函数
for i in range(5):          # 0-4
    print(i)

for i in range(2, 10, 2):  # 2, 4, 6, 8
    print(i)

# AI开发中的for循环应用
# 遍历训练批次
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 训练逻辑
        pass
```

#### while循环

```python
# 基本while循环
count = 0
while count < 5:
    print(count)
    count += 1

# while循环配合else
count = 0
while count < 5:
    print(count)
    count += 1
else:
    print("Loop completed")

# AI开发中的while循环应用
# 生成对抗网络（GAN）中的训练循环
while epoch < max_epochs:
    # 训练生成器和判别器
    # 更新epoch计数
    epoch += 1
```

#### 循环控制语句

```python
# break：跳出当前循环
for i in range(10):
    if i == 5:
        break
    print(i)  # 输出0-4

# continue：跳过当前迭代，继续下一次循环
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)  # 输出1, 3, 5, 7, 9

# pass：占位符，不执行任何操作
for i in range(10):
    if i < 5:
        pass  # 未来可能添加代码
    else:
        print(i)

# AI开发中的循环控制
# 在模型预测时跳过异常样本
for sample in data_samples:
    try:
        prediction = model.predict(sample)
    except Exception as e:
        print(f"Error processing sample: {e}")
        continue  # 跳过当前样本，继续处理下一个
```

## 4. 函数编程

### 4.1 函数定义与调用

```python
def greet(name):
    """
    向指定名称的人打招呼
    
    参数:
        name (str): 要打招呼的人的名字
        
    返回:
        str: 打招呼的字符串
    """
    return f"Hello, {name}!"

# 调用函数
result = greet("Alice")
print(result)  # "Hello, Alice!"

# AI开发中的函数设计
# 定义模型评估函数
def evaluate_model(model, test_data, test_labels):
    """
    评估模型在测试集上的性能
    
    参数:
        model: 要评估的模型
        test_data: 测试数据
        test_labels: 测试标签
        
    返回:
        dict: 包含准确率、精确率、召回率等指标的字典
    """
    # 评估逻辑
    return metrics
```

### 4.2 函数参数

```python
# 位置参数
def add(a, b):
    return a + b

# 默认参数
def power(base, exponent=2):
    return base ** exponent

# 关键字参数
print(power(3, exponent=3))  # 27
print(power(exponent=3, base=2))  # 8

# 可变参数（*args）
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4))  # 10

# 关键字可变参数（**kwargs）
def print_kwargs(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_kwargs(name="Alice", age=30)

# AI开发中的函数参数应用
# 定义训练函数，支持多种优化器
def train_model(model, train_data, train_labels, optimizer="adam", learning_rate=0.001, epochs=100):
    # 训练逻辑
    pass
```

### 4.3 匿名函数（lambda）

```python
# 基本语法：lambda 参数: 表达式
double = lambda x: x * 2
print(double(5))  # 10

# 结合内置函数使用
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))  # [1, 4, 9, 16, 25]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]

sum_result = reduce(lambda x, y: x + y, numbers)  # 15

# AI开发中的lambda应用
# 数据预处理中的匿名函数
import pandas as pd
df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
df['normalized'] = df['feature1'].apply(lambda x: (x - df['feature1'].mean()) / df['feature1'].std())
```

### 4.4 高阶函数

```python
# 高阶函数：接受函数作为参数或返回函数
def apply_func(func, x):
    return func(x)

result = apply_func(lambda x: x * 2, 5)  # 10

# 返回函数
def create_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

double = create_multiplier(2)
triple = create_multiplier(3)
print(double(5))  # 10
print(triple(5))  # 15

# AI开发中的高阶函数应用
# 使用高阶函数实现模型训练回调
def create_early_stopping(patience=5):
    best_loss = float('inf')
    patience_counter = 0
    
    def early_stop(current_loss):
        nonlocal best_loss, patience_counter
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            return False
        else:
            patience_counter += 1
            return patience_counter >= patience
    
    return early_stop
```

## 5. 面向对象编程

### 5.1 类与对象

```python
class Person:
    """人员类"""
    
    # 类属性
    species = "Homo sapiens"
    
    # 初始化方法
    def __init__(self, name, age):
        # 实例属性
        self.name = name
        self.age = age
    
    # 实例方法
    def greet(self):
        return f"Hello, my name is {self.name} and I'm {self.age} years old"
    
    # 类方法
    @classmethod
    def from_birth_year(cls, name, birth_year):
        current_year = 2026
        age = current_year - birth_year
        return cls(name, age)
    
    # 静态方法
    @staticmethod
    def is_adult(age):
        return age >= 18

# 创建对象
person1 = Person("Alice", 30)
person2 = Person.from_birth_year("Bob", 1990)

# 调用方法
print(person1.greet())  # "Hello, my name is Alice and I'm 30 years old"
print(Person.is_adult(25))  # True

# AI开发中的类设计
# 定义神经网络层类
class DenseLayer:
    """全连接神经网络层"""
    
    def __init__(self, input_dim, output_dim, activation="relu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        # 初始化权重和偏置
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()
    
    def _initialize_weights(self):
        # 权重初始化逻辑
        pass
    
    def _initialize_biases(self):
        # 偏置初始化逻辑
        pass
    
    def forward(self, inputs):
        # 前向传播计算
        pass
    
    def backward(self, gradients):
        # 反向传播计算
        pass
```

### 5.2 继承与多态

```python
# 父类
class Animal:
    def __init__(self, name):
        self.name = name
    
    def make_sound(self):
        raise NotImplementedError("子类必须实现此方法")

# 子类继承
class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

# 多态
animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(f"{animal.name} says: {animal.make_sound()}")

# AI开发中的继承应用
# 基于基础模型类创建特定模型
class BaseModel:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        # 基础前向传播逻辑
        pass
    
    def train(self, train_data, train_labels):
        # 基础训练逻辑
        pass

class ClassificationModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    def calculate_loss(self, predictions, targets):
        # 分类损失计算
        pass
    
    def predict(self, inputs):
        # 分类预测逻辑
        pass

class RegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
    
    def calculate_loss(self, predictions, targets):
        # 回归损失计算
        pass
```

### 5.3 特殊方法（魔术方法）

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # 加法运算
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    # 乘法运算
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    # 字符串表示
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    # 长度计算
    def __len__(self):
        return int((self.x ** 2 + self.y ** 2) ** 0.5)

v1 = Vector(3, 4)
v2 = Vector(1, 2)
v3 = v1 + v2  # Vector(4, 6)
v4 = v1 * 2    # Vector(6, 8)

print(v3)  # "Vector(4, 6)"
print(len(v1))  # 5

# AI开发中的特殊方法应用
# 自定义张量类
class Tensor:
    def __init__(self, data, shape):
        self.data = data
        self.shape = shape
    
    def __matmul__(self, other):
        # 实现矩阵乘法 @ 运算符
        pass
    
    def __getitem__(self, index):
        # 实现索引访问
        pass
    
    def __setitem__(self, index, value):
        # 实现索引赋值
        pass
```

## 6. 模块与包管理

### 6.1 模块导入

```python
# 导入整个模块
import math
print(math.pi)  # 3.141592653589793

# 导入特定函数或变量
from math import pi, sqrt
print(sqrt(16))  # 4.0

# 导入所有内容（不推荐）
from math import *

# 导入并别名
import numpy as np
import pandas as pd

# AI开发中常用的模块导入
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

### 6.2 自定义模块

```python
# 创建自定义模块：my_module.py
# def add(a, b):
#     return a + b
# 
# def subtract(a, b):
#     return a - b

# 导入自定义模块
import my_module
print(my_module.add(5, 3))  # 8

# 从自定义模块导入特定函数
from my_module import subtract
print(subtract(5, 3))  # 2

# AI开发中自定义模块的应用
# 创建数据预处理模块：data_preprocessing.py
# 包含数据清洗、特征工程等函数
# 然后在主训练脚本中导入使用
```

### 6.3 包管理

```python
# 包是包含多个模块的目录，必须包含__init__.py文件
# my_package/
#     __init__.py
#     module1.py
#     module2.py

# 导入包
import my_package
from my_package import module1
from my_package.module2 import function_name

# __init__.py的作用
# 1. 标识目录为Python包
# 2. 可以在其中定义包级别的变量和函数
# 3. 控制包的导入行为

# AI开发中的包结构示例
# project/
#     __init__.py
#     data/
#         __init__.py
#         loader.py
#         preprocessor.py
#     models/
#         __init__.py
#         base_model.py
#         cnn_model.py
#         rnn_model.py
#     training/
#         __init__.py
#         trainer.py
#         optimizer.py
#     utils/
#         __init__.py
#         metrics.py
#         visualization.py
```

### 6.4 使用pip管理包

```bash
# 安装包
pip install numpy
pip install tensorflow==2.12.0  # 安装特定版本

# 升级包
pip install --upgrade numpy

# 卸载包
pip uninstall numpy

# 列出已安装的包
pip list
pip freeze  # 生成requirements.txt格式的输出

# 从requirements.txt安装
pip install -r requirements.txt

# AI开发中requirements.txt示例：
# numpy>=1.21.0
# pandas>=1.3.0
# tensorflow>=2.10.0
# torch>=1.12.0
# scikit-learn>=1.0.0
# jupyter>=1.0.0
```

## 7. 异常处理

### 7.1 基本异常处理

```python
# try-except结构
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Error: Division by zero")

# 捕获多个异常
try:
    with open("nonexistent.txt", "r") as f:
        content = f.read()
except FileNotFoundError:
    print("Error: File not found")
except PermissionError:
    print("Error: Permission denied")
except Exception as e:  # 捕获所有其他异常
    print(f"Error: {e}")

# try-except-else-finally
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Division by zero")
else:
    print(f"Result: {result}")  # 只有在没有异常时执行
finally:
    print("This will always execute")  # 无论是否有异常都会执行

# AI开发中的异常处理
# 处理模型训练中的异常
try:
    model.train()
    for batch in train_loader:
        # 训练逻辑
        pass
except KeyboardInterrupt:
    print("Training interrupted by user")
    # 保存当前模型状态
    torch.save(model.state_dict(), "interrupted_model.pth")
except RuntimeError as e:
    print(f"Runtime error during training: {e}")
    # 处理内存不足等问题
```

### 7.2 自定义异常

```python
# 自定义异常类
class ModelTrainingError(Exception):
    """模型训练过程中发生的异常"""
    pass

class DataFormatError(Exception):
    """数据格式错误"""
    pass

# 使用自定义异常
def validate_data(data):
    if not isinstance(data, dict):
        raise DataFormatError("Data must be a dictionary")
    if "features" not in data:
        raise DataFormatError("Data must contain 'features' key")
    return True

# AI开发中自定义异常的应用
# 模型评估时抛出异常
def evaluate_model(model, data):
    try:
        validate_data(data)
        # 评估逻辑
    except DataFormatError as e:
        raise ModelTrainingError(f"Evaluation failed: {e}")
```

## 8. 文件操作

### 8.1 文件的打开与关闭

```python
# 基本文件操作
# 方式1：显式关闭文件
file = open("example.txt", "w")
file.write("Hello, Python!")
file.close()

# 方式2：使用with语句（推荐，自动关闭文件）
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

# 文件打开模式
# r: 只读（默认）
# w: 写入（覆盖现有文件）
# a: 追加写入
# x: 排他性创建（如果文件已存在则失败）
# b: 二进制模式
# t: 文本模式（默认）
# +: 读写模式

# AI开发中的文件操作
# 读取训练数据
with open("train_data.txt", "r") as f:
    train_data = [line.strip() for line in f]

# 保存模型参数
import json
model_params = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100}
with open("model_params.json", "w") as f:
    json.dump(model_params, f, indent=4)
```

### 8.2 文件读写操作

```python
# 读取文件
with open("example.txt", "r") as f:
    # 读取全部内容
    content = f.read()
    
    # 读取一行
    line1 = f.readline()
    
    # 读取所有行
    lines = f.readlines()

# 写入文件
with open("output.txt", "w") as f:
    # 写入字符串
    f.write("First line\n")
    
    # 写入多行
    lines = ["Second line\n", "Third line\n"]
    f.writelines(lines)

# AI开发中的文件读写
# 读取CSV数据（使用Pandas更高效，但这里展示基础方法）
data = []
with open("data.csv", "r") as f:
    header = f.readline().strip().split(",")
    for line in f:
        values = line.strip().split(",")
        data.append(dict(zip(header, values)))

# 保存训练日志
with open("training_log.txt", "a") as f:
    f.write(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}\n")
```

### 8.3 目录操作

```python
import os

# 创建目录
os.makedirs("outputs/models", exist_ok=True)  # exist_ok=True避免目录已存在时出错

# 查看当前目录
current_dir = os.getcwd()

# 列出目录内容
files = os.listdir(".")

# 检查文件/目录是否存在
if os.path.exists("data"):
    print("Data directory exists")

# 检查是否为文件/目录
os.path.isfile("example.txt")
os.path.isdir("outputs")

# 获取文件大小
file_size = os.path.getsize("example.txt")

# 获取绝对路径
abs_path = os.path.abspath("example.txt")

# 路径拼接（跨平台兼容）
data_path = os.path.join("datasets", "train", "images")

# AI开发中的目录操作
# 创建实验目录
experiment_dir = f"experiments/exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(experiment_dir, exist_ok=True)

# 保存实验结果
results_path = os.path.join(experiment_dir, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f)
```

## 9. 常用标准库

### 9.1 数据处理相关库

#### math - 数学函数

```python
import math

# 常用数学函数
math.pi          # π
math.e           # 自然常数e
math.sqrt(16)    # 平方根
math.pow(2, 3)   # 幂运算
math.sin(math.pi/2)  # 三角函数
math.log(10)     # 自然对数
math.log10(100)  # 对数（底数10）
math.ceil(3.14)  # 向上取整
math.floor(3.99) # 向下取整

# AI开发中的应用
# 计算欧氏距离
def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
```

#### random - 随机数生成

```python
import random

# 生成随机整数
random.randint(1, 100)  # 1-100之间的随机整数

# 生成随机浮点数
random.random()  # 0-1之间的随机浮点数
random.uniform(1.0, 10.0)  # 指定范围的随机浮点数

# 随机选择
random.choice([1, 2, 3, 4, 5])  # 从列表中随机选择一个元素

# 随机打乱
numbers = [1, 2, 3, 4, 5]
random.shuffle(numbers)

# AI开发中的应用
# 随机划分训练集和验证集
def split_train_val(data, val_ratio=0.2):
    random.shuffle(data)
    val_size = int(len(data) * val_ratio)
    return data[val_size:], data[:val_size]
```

#### datetime - 日期和时间

```python
from datetime import datetime, timedelta

# 获取当前时间
now = datetime.now()
print(now)  # 2026-01-25 14:30:00.123456

# 格式化时间
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print(formatted)  # "2026-01-25 14:30:00"

# 解析时间字符串
date_str = "2026-01-25"
date_obj = datetime.strptime(date_str, "%Y-%m-%d")

# 时间运算
tomorrow = now + timedelta(days=1)
yesterday = now - timedelta(days=1)

# AI开发中的应用
# 记录训练时间
start_time = datetime.now()
# 训练逻辑
train_model(model, train_data, train_labels)
end_time = datetime.now()
training_duration = end_time - start_time
print(f"Training time: {training_duration}")
```

### 9.2 数据结构与算法相关库

#### collections - 扩展数据结构

```python
from collections import Counter, defaultdict, deque, namedtuple

# Counter：计数
fruits = ["apple", "banana", "apple", "orange", "banana", "apple"]
fruit_count = Counter(fruits)
print(fruit_count)  # Counter({'apple': 3, 'banana': 2, 'orange': 1})

# defaultdict：默认值字典
word_counts = defaultdict(int)
words = ["hello", "world", "hello"]
for word in words:
    word_counts[word] += 1

# deque：双端队列（高效的插入和删除）
queue = deque([1, 2, 3])
queue.append(4)      # 右侧添加
queue.appendleft(0)  # 左侧添加
queue.pop()          # 右侧弹出
queue.popleft()      # 左侧弹出

# namedtuple：命名元组
Point = namedtuple("Point", ["x", "y"])
p = Point(10, 20)
print(p.x, p.y)  # 10 20

# AI开发中的应用
# 统计类别分布
from collections import Counter
labels = [0, 1, 0, 1, 1, 2, 0, 2, 2, 2]
label_distribution = Counter(labels)
print(label_distribution)  # Counter({2: 4, 1: 3, 0: 3})
```

#### itertools - 迭代器工具

```python
from itertools import product, permutations, combinations, chain, cycle, repeat

# product：笛卡尔积
for x, y in product([1, 2], [3, 4]):
    print(x, y)  # (1,3), (1,4), (2,3), (2,4)

# permutations：排列
for p in permutations([1, 2, 3], 2):
    print(p)  # (1,2), (1,3), (2,1), (2,3), (3,1), (3,2)

# combinations：组合
for c in combinations([1, 2, 3], 2):
    print(c)  # (1,2), (1,3), (2,3)

# chain：链接多个迭代器
combined = chain([1, 2, 3], [4, 5, 6])
print(list(combined))  # [1, 2, 3, 4, 5, 6]

# AI开发中的应用
# 生成超参数组合
hyperparams = {
    "learning_rate": [0.001, 0.01],
    "batch_size": [32, 64],
    "optimizer": ["adam", "sgd"]
}

from itertools import product
param_combinations = list(product(*hyperparams.values()))
for params in param_combinations:
    param_dict = dict(zip(hyperparams.keys(), params))
    print(param_dict)
```

### 9.3 其他常用标准库

#### json - JSON数据处理

```python
import json

# 序列化（Python对象转JSON）
data = {
    "name": "Alice",
    "age": 30,
    "is_student": False,
    "courses": ["Math", "Computer Science"]
}

json_str = json.dumps(data, indent=4)  # 格式化输出
print(json_str)

# 反序列化（JSON转Python对象）
json_str = '{"name": "Bob", "age": 25}'
data = json.loads(json_str)
print(data["name"])  # "Bob"

# AI开发中的应用
# 保存和加载模型配置
model_config = {
    "model_name": "CNN",
    "input_shape": [28, 28, 1],
    "output_shape": 10,
    "layers": [
        {"type": "Conv2D", "filters": 32, "kernel_size": [3, 3], "activation": "relu"},
        {"type": "MaxPooling2D", "pool_size": [2, 2]},
        {"type": "Flatten"},
        {"type": "Dense", "units": 128, "activation": "relu"},
        {"type": "Dense", "units": 10, "activation": "softmax"}
    ]
}

# 保存配置
with open("model_config.json", "w") as f:
    json.dump(model_config, f, indent=4)

# 加载配置
with open("model_config.json", "r") as f:
    loaded_config = json.load(f)
```

#### os - 操作系统接口

```python
import os

# 目录操作
os.getcwd()          # 获取当前工作目录
os.chdir("/path/to/directory")  # 切换目录
os.listdir(".")       # 列出目录内容
os.makedirs("dir/subdir", exist_ok=True)  # 创建目录树
os.remove("file.txt")  # 删除文件
os.rmdir("directory")  # 删除目录（必须为空）
os.removedirs("dir/subdir")  # 递归删除空目录

# 文件和路径操作
os.path.exists("file.txt")  # 检查路径是否存在
os.path.isfile("file.txt")  # 检查是否为文件
os.path.isdir("directory")  # 检查是否为目录
os.path.join("dir", "file.txt")  # 拼接路径
os.path.abspath("file.txt")  # 获取绝对路径
os.path.basename("/path/to/file.txt")  # 获取文件名
os.path.dirname("/path/to/file.txt")  # 获取目录名

# AI开发中的应用
# 遍历数据目录
def get_image_paths(data_dir):
    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return image_paths
```

## 10. 高级Python特性

### 10.1 列表推导式

```python
# 基本语法：[expression for item in iterable if condition]
numbers = [1, 2, 3, 4, 5]
squared = [x ** 2 for x in numbers]  # [1, 4, 9, 16, 25]
even_squared = [x ** 2 for x in numbers if x % 2 == 0]  # [4, 16]

# 嵌套列表推导式
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
transposed = [[row[i] for row in matrix] for i in range(3)]  # [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

# AI开发中的应用
# 预处理图像数据
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
processed_images = [preprocess_image(img) for img in images if is_valid_image(img)]

# 生成标签列表
labels = [1 if img_path.contains("cat") else 0 for img_path in image_paths]
```

### 10.2 字典推导式

```python
# 基本语法：{key_expression: value_expression for item in iterable if condition}
numbers = [1, 2, 3, 4, 5]
squared_dict = {x: x ** 2 for x in numbers}  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# 从两个列表创建字典
keys = ["a", "b", "c"]
values = [1, 2, 3]
dictionary = {k: v for k, v in zip(keys, values)}  # {"a": 1, "b": 2, "c": 3}

# AI开发中的应用
# 创建类别映射
class_names = ["cat", "dog", "bird"]
class_to_idx = {name: idx for idx, name in enumerate(class_names)}  # {"cat": 0, "dog": 1, "bird": 2}

# 统计特征频率
features = ["f1", "f2", "f1", "f3", "f2", "f1"]
feature_counts = {f: features.count(f) for f in set(features)}  # {"f1": 3, "f2": 2, "f3": 1}
```

### 10.3 生成器

```python
# 生成器表达式
numbers = [1, 2, 3, 4, 5]
squared_gen = (x ** 2 for x in numbers)  # 生成器表达式
print(list(squared_gen))  # [1, 4, 9, 16, 25]

# 生成器函数（使用yield关键字）
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 使用生成器
for num in fibonacci(10):
    print(num, end=" ")  # 0 1 1 2 3 5 8 13 21 34

# AI开发中的应用
# 大数据集生成器
def data_generator(file_path, batch_size=32):
    """
    生成器函数，用于逐批加载大型数据集
    """
    batch = []
    with open(file_path, "r") as f:
        for line in f:
            # 处理每行数据
            data = process_line(line)
            batch.append(data)
            if len(batch) == batch_size:
                yield batch
                batch = []
    # 处理剩余数据
    if batch:
        yield batch

# 使用生成器训练模型
for batch in data_generator("large_dataset.txt", batch_size=64):
    model.train_on_batch(batch)
```

### 10.4 装饰器

```python
# 简单装饰器
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

# 使用装饰器
@timing_decorator
def slow_function():
    time.sleep(1)
    return "Done"

slow_function()  # 输出：slow_function took 1.0001 seconds to execute

# 带参数的装饰器
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(n):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(3)
def say_hello(name):
    return f"Hello, {name}!"

print(say_hello("Alice"))  # ["Hello, Alice!", "Hello, Alice!", "Hello, Alice!"]

# AI开发中的装饰器应用
# 模型评估装饰器
def evaluate_model_decorator(func):
    def wrapper(model, data, labels, *args, **kwargs):
        print("Evaluating model performance...")
        result = func(model, data, labels, *args, **kwargs)
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Loss: {result['loss']:.4f}")
        return result
    return wrapper

@evaluate_model_decorator
def evaluate(model, data, labels):
    # 评估逻辑
    return {"accuracy": 0.95, "loss": 0.12}
```

## 11. Python最佳实践与AI开发技巧

### 11.1 代码风格与规范

```python
# 遵循PEP 8风格指南
# 1. 使用4个空格缩进
# 2. 行长度不超过79个字符
# 3. 使用小写字母加下划线命名变量和函数
# 4. 使用驼峰命名法命名类
# 5. 函数和类之间空两行
# 6. 导入语句按顺序排列：标准库、第三方库、本地库

# 安装和使用flake8检查代码风格
# pip install flake8
# flake8 your_script.py

# AI开发中代码组织建议
# 1. 将不同功能模块分离到不同文件
# 2. 使用清晰的目录结构
# 3. 为函数和类添加详细的文档字符串
# 4. 使用类型提示提高代码可读性和可维护性
```

### 11.2 性能优化

```python
# 1. 使用列表推导式替代循环
# 低效
result = []
for i in range(1000):
    result.append(i * 2)

# 高效
result = [i * 2 for i in range(1000)]

# 2. 避免在循环中使用全局变量
# 低效
global_var = 10
def slow_func():
    result = 0
    for i in range(1000):
        result += global_var
    return result

# 高效
def fast_func():
    local_var = global_var  # 将全局变量转为局部变量
    result = 0
    for i in range(1000):
        result += local_var
    return result

# 3. 使用生成器处理大数据
# 低效：一次性加载所有数据到内存
large_list = [i for i in range(1000000)]
for item in large_list:
    process(item)

# 高效：逐次生成数据，节省内存
def large_generator():
    for i in range(1000000):
        yield i

for item in large_generator():
    process(item)

# AI开发中的性能优化
# 1. 使用GPU加速（TensorFlow、PyTorch）
# 2. 批量处理数据
# 3. 使用高效的数据结构（NumPy数组替代Python列表）
# 4. 避免在训练循环中进行不必要的计算
# 5. 使用混合精度训练减少内存使用和提高计算速度
```

### 11.3 调试技巧

```python
# 1. 使用print语句进行简单调试
x = 10
print(f"x = {x}")

# 2. 使用断点调试
# 在IDE中设置断点，或使用pdb库
import pdb

def buggy_function():
    x = 10
    y = 0
    pdb.set_trace()  # 设置断点
    result = x / y
    return result

# 3. 使用assert语句验证假设
assert x > 0, "x must be positive"

# 4. 使用logging模块进行日志记录
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")

# AI开发中的调试技巧
# 1. 使用TensorBoard可视化训练过程
# 2. 检查模型输出形状和数据类型
# 3. 监控GPU/CPU使用率和内存占用
# 4. 使用model.summary()查看模型结构
# 5. 对小样本数据进行测试，验证模型是否正常工作
```

## 12. 总结

Python作为AI开发的首选语言，其基础知识是AI开发者必备的技能。本文系统介绍了Python的核心知识点，包括：

- Python语言基础与语法规则
- 数据类型与数据结构
- 控制流与循环
- 函数编程与面向对象编程
- 模块与包管理
- 异常处理
- 文件操作
- 常用标准库
- 高级特性（列表推导式、生成器、装饰器等）
- 最佳实践与性能优化

每个知识点都结合了AI开发场景，提供了实用的代码示例和应用案例，旨在帮助AI开发者系统掌握Python编程技能，提高开发效率，更好地应用于AI模型开发、数据处理、模型部署等工作中。

随着AI技术的不断发展，Python生态也在持续完善。AI开发者应保持学习状态，不断掌握新的Python库和技术，以适应快速变化的AI开发需求。

## 更新记录

- 2026-01-25：创建文件，系统介绍Python基础知识，结合AI开发场景

## 参考资料

- [Python官方文档](https://docs.python.org/3/)
- [PEP 8风格指南](https://www.python.org/dev/peps/pep-0008/)
- [NumPy官方文档](https://numpy.org/doc/)
- [TensorFlow官方文档](https://www.tensorflow.org/)
- [PyTorch官方文档](https://pytorch.org/docs/stable/)

---

> 本文件为AI开发人员提供了系统全面的Python基础知识参考，内容涵盖从入门到进阶的学习需求，结合实际AI开发场景，帮助开发者高效掌握Python编程技能。