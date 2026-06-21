# TensorFlow 转换方法

> **标签**: #tensorflow #tensorflow-keras #conversion
> **相关链接**: [[04-跨框架转换/框架对比分析]]

## 概述

TensorFlow 模型转换为 ONNX 主要使用 `tf2onnx` 工具，支持 TensorFlow 1.x 和 2.x 版本。对于 Keras 模型，通常先转换为 SavedModel 格式，再进行 ONNX 导出。

## 环境准备

```bash
# 安装 tf2onnx
pip install tf2onnx

# TensorFlow（根据需求选择）
pip install tensorflow        # CPU 版本
pip install tensorflow-gpu   # GPU 版本（TF 1.x）
pip install tensorflow[and-cuda]  # TF 2.x GPU

# 可选：ONNX Runtime（用于验证）
pip install onnxruntime
```

**版本兼容性表**:

| TensorFlow | 推荐 tf2onnx | ONNX opset |
|-----------|-------------|-----------|
| 2.13+     | 1.14+       | 15        |
| 2.10-2.12 | 1.13-1.14   | 14-15     |
| 2.5-2.9   | 1.10-1.12   | 12-13     |
| 1.15      | 1.5-1.9     | 9-10      |

## 转换方法一：命令行工具

### 基本用法

```bash
# TensorFlow 2.x / Keras 模型
python -m tf2onnx.convert \
    --saved-model /path/to/saved_model \
    --output model.onnx

# 指定 opset 版本
python -m tf2onnx.convert \
    --saved-model saved_model_dir \
    --output model.onnx \
    --opset 13

# H5 格式 Keras 模型
python -m tf2onnx.convert \
    --keras model.h5 \
    --output model.onnx
```

### 常用命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--saved-model` | SavedModel 目录 | `--saved-model ./model` |
| `--keras` | Keras H5 文件路径 | `--keras model.h5` |
| `--checkpoint` | TensorFlow checkpoint | `--checkpoint model.ckpt` |
| `--graphdef` | GraphDef 文件 | `--graphdef model.pb` |
| `--output` | 输出 ONNX 文件 | `--output model.onnx` |
| `--opset` | ONNX opset 版本 | `--opset 13` |
| `--verbose` | 详细日志输出 | `--verbose` |
| `--inputs` | 指定输入节点 | `--inputs input:0` |
| `--outputs` | 指定输出节点 | `--outputs output:0` |
| `--extra_opset` | 额外 opset 域 | `--extra_opset ai.onnx:12` |

### tf2onnx 命令行参数详解

```bash
# 查看所有参数
python -m tf2onnx -h

# 完整示例：转换 MobileNetV2
python -m tf2onnx.convert \
    --saved-model mobilenet_v2/ \
    --output mobilenet_v2.onnx \
    --opset 13 \
    --verbose \
    --inputs input:0 \
    --outputs MobilenetV2/Predictions/Reshape_1:0
```

## 转换方法二：Python API

### 基本转换

```python
import tf2onnx
import tensorflow as tf
import numpy as np

# 方式1: 从 SavedModel 转换
model_proto, _ = tf2onnx.convert.from_saved_model(
    saved_model_dir='saved_model_dir',
    input_signature=None,  # 可指定输入签名
    opset=13,
    custom_ops=None,
    custom_op_handlers=None,
    custom_rewriter=None,
    extra_opset=None,
    shape_override=None,
    verbose=False
)

# 保存 ONNX 模型
with open('model.onnx', 'wb') as f:
    f.write(model_proto.SerializeToString())

print("✓ 转换完成")
```

### 从 Keras 模型直接转换

```python
import tensorflow as tf
import tf2onnx

# 1. 加载 Keras 模型
model = tf.keras.applications.MobileNetV2()
model.compile()

# 2. 转换
input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input')]

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    custom_objects=None  # 自定义层需要注册
)

# 3. 保存
onnx_model_path = 'mobilenet_v2.onnx'
with open(onnx_model_path, 'wb') as f:
    f.write(model_proto.SerializeToString())

print(f"✓ 模型已保存到 {onnx_model_path}")
```

## SavedModel 转换工作流

### 完整流程

```python
import tensorflow as tf
import tf2onnx
import os

# 步骤1: 准备 Keras 模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 步骤2: 导出为 SavedModel
saved_model_dir = 'my_model'
model.save(saved_model_dir, save_format='tf')
print(f"✓ SavedModel 已保存到 {saved_model_dir}")

# 步骤3: 转换为 ONNX
model_proto, _ = tf2onnx.convert.from_saved_model(
    saved_model_dir,
    input_names=['input:0'],    # SavedModel 的默认输入名
    output_names=['output:0'],  # 输出节点名
    opset=13
)

onnx_path = 'model.onnx'
with open(onnx_path, 'wb') as f:
    f.write(model_proto.SerializeToString())

print(f"✓ ONNX 模型已保存到 {onnx_path}")
```

### 高级工作流：控制输入/输出名称

```python
import tensorflow as tf
import tf2onnx

# 使用自定义签名
class CustomModel(tf.keras.Model):
    def call(self, inputs):
        return {'logits': self._model(inputs), 'features': self._features(inputs)}

model = CustomModel()

# 指定具体的输入输出
input_names = ['images:0']
output_names = ['logits:0', 'features:0']

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32, name='images')],
    opset=13,
    custom_objects=None,
    # 转换后重命名节点
    output_path='model.onnx'
)

# 如果需要指定输入输出，使用:
# model_proto, _ = tf2onnx.convert.from_keras(
#     model,
#     input_signature=...,
#     opset=13,
#     custom_objects=None,
#     # 转换后需要通过 onnx 工具重命名
# )
```

## opset 版本选择策略

### 根据 TensorFlow 版本选择

```python
import tensorflow as tf

tf_version = tf.__version__
major, minor = map(int, tf_version.split('.')[:2])

# 自动选择 opset
if major >= 2 and minor >= 13:
    recommended_opset = 15
elif major >= 2 and minor >= 10:
    recommended_opset = 13
elif major >= 2 and minor >= 7:
    recommended_opset = 11
else:
    recommended_opset = 10

print(f"TensorFlow {tf_version} 推荐 opset: {recommended_opset}")
```

### 兼容性权衡

| 考虑因素 | 低 opset | 高 opset |
|---------|---------|---------|
| 推理引擎支持 | ✓ 广泛兼容 | ✗ 需要较新 ONNX Runtime |
| 新算子支持 | ✗ 有限 | ✓ 完整 |
| 优化程度 | 一般 | ✓ 更好 |
| 生产稳定性 | ✓ 成熟 | 较新可能有 bug |

**推荐策略**:
1. 目标部署环境已知 → 选择其支持的最高 opset
2. 多平台部署 → 选择 opset 11-13 作为基线
3. 需要最新算子 → opset 14-15

## 示例：CNN 模型转换（ResNet50）

```python
import tensorflow as tf
import tf2onnx
import numpy as np

# 1. 加载预训练模型
model = tf.keras.applications.ResNet50(
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# 2. 导出为 SavedModel
saved_model_dir = 'resnet50_saved_model'
model.save(saved_model_dir, save_format='tf')
print(f"SavedModel 保存至: {saved_model_dir}")

# 3. 转换为 ONNX（指定输入输出）
model_proto, _ = tf2onnx.convert.from_saved_model(
    saved_model_dir,
    input_names=['input_1:0'],      # Keras 默认输入名
    output_names=['predictions:0'], # 输出层名
    opset=13,
    # 动态 batch（通过 shape_override）
    shape_override={
        'input_1:0': [None, 224, 224, 3]
    }
)

# 4. 保存 ONNX
onnx_path = 'resnet50.onnx'
with open(onnx_path, 'wb') as f:
    f.write(model_proto.SerializeToString())

# 5. 验证
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path)

# 随机输入测试
test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
outputs = sess.run(None, {'input_1:0': test_input})
print(f"输出形状: {outputs[0].shape}")

print("✓ ResNet50 转换完成")
```

## 示例：RNN 模型转换（LSTM）

```python
import tensorflow as tf
import tf2onnx
import numpy as np

# 1. 创建 LSTM 模型
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(None, 128)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 构建模型
dummy_input = np.random.randn(2, 10, 128).astype(np.float32)
_ = model(dummy_input)

# 2. 保存 SavedModel
saved_model_dir = 'lstm_model'
model.save(saved_model_dir, save_format='tf')

# 3. 转换为 ONNX
# RNN 动态序列长度需要特殊处理
input_names = ['lstm_input:0']
output_names = ['dense_output:0']

model_proto, _ = tf2onnx.convert.from_saved_model(
    saved_model_dir,
    input_names=input_names,
    output_names=output_names,
    opset=13
)

with open('lstm_model.onnx', 'wb') as f:
    f.write(model_proto.SerializeToString())

print("✓ LSTM 模型转换完成")
```

## 故障排查技巧

### 问题1: 找不到输入/输出节点

**错误信息**: `ValueError: Cannot find input...`

**诊断方法**:
```python
import tensorflow as tf
import tf2onnx

# 查看 SavedModel 的签名
loaded = tf.saved_model.load('saved_model_dir')
print(list(loaded.signatures.keys()))  # 查看默认签名

# 查看具体的输入输出结构
infer = loaded.signatures['serving_default']
print("输入:")
for name, tensor in infer.structured_input_signature[1].items():
    print(f"  {name}: {tensor.shape}, {tensor.dtype}")
print("输出:")
for name, tensor in infer.structured_outputs.items():
    print(f"  {name}: {tensor.shape}, {tensor.dtype}")
```

**解决方案**:
```python
# 使用正确的输入输出名称
model_proto, _ = tf2onnx.convert.from_saved_model(
    'saved_model_dir',
    input_names=['input_1:0'],   # 从上述输出中获取准确名称
    output_names=['Identity:0']  # 通常是 Identity 或预测层
)
```

### 问题2: 转换后的模型输出错误

```python
# 使用 tf2onnx 的诊断工具
from tf2onnx import utils

# 打印模型信息
utils.print_model_graph('saved_model_dir')

# 使用详细模式
model_proto, _ = tf2onnx.convert.from_saved_model(
    'saved_model_dir',
    verbose=True  # 打印详细信息
)
```

### 问题3: RNN/LSTM 转换失败

**常见症状**: 序列维度处理异常

**解决方案**:
```python
# 在转换前确保模型已经 build
# 方式1: 用实际数据运行一次
dummy_input = np.random.randn(1, 10, 128).astype(np.float32)
_ = model(dummy_input)

# 方式2: 显式 build
model.build(input_shape=(None, None, 128))

# 然后再导出
model.save('saved_model')
```

### 问题4: 自定义层转换失败

```python
import tensorflow as tf
import tf2onnx

# 自定义层
class CustomLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs * 2

# 注册自定义对象
custom_objects = {'CustomLayer': CustomLayer}

# 转换时使用
model = tf.keras.models.load_model('model_with_custom.h5',
                                   custom_objects=custom_objects)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[tf.TensorSpec([None, 10], tf.float32)],
    custom_objects=custom_objects  # 传递自定义对象
)
```

## 从 Checkpoint 转换

```python
import tensorflow as tf
import tf2onnx

# 方式1: 使用 checkpoint 文件
model_proto, _ = tf2onnx.convert.from_checkpoint(
    checkpoint_path='model.ckpt',
    output_path='model.onnx',
    inputs=['input:0'],
    outputs=['output:0'],
    opset=13
)

# 方式2: 手动加载权重
# 先构建模型架构
model = create_model()  # 自定义函数
model.load_weights('model.ckpt')

# 保存为 SavedModel
model.save('temp_saved_model', save_format='tf')

# 再转换
tf2onnx.convert.from_saved_model('temp_saved_model', output_path='model.onnx')
```

## 转换质量验证

```python
import tensorflow as tf
import tf2onnx
import numpy as np
import onnxruntime as ort

def verify_conversion(saved_model_path, onnx_path, test_inputs):
    """验证 TensorFlow 和 ONNX 输出一致性"""

    # TensorFlow 推理
    tf_model = tf.saved_model.load(saved_model_path)
    infer = tf_model.signatures['serving_default']
    tf_outputs = infer(**test_inputs)

    # ONNX 推理
    sess = ort.InferenceSession(onnx_path)
    onnx_inputs = {k: v.numpy() for k, v in test_inputs.items()}
    onnx_outputs = sess.run(None, onnx_inputs)

    # 对比
    print("验证结果:")
    for (tf_name, tf_out), onnx_out in zip(tf_outputs.items(), onnx_outputs):
        diff = np.abs(tf_out.numpy() - onnx_out).max()
        print(f"  {tf_name}: 最大误差 = {diff:.2e}")

        if diff > 1e-4:
            print(f"  ⚠ 警告: {tf_name} 输出差异较大！")
        else:
            print(f"  ✓ 通过")

# 使用示例
test_inputs = {
    'input_1:0': tf.random.normal((1, 224, 224, 3))
}
verify_conversion('saved_model_dir', 'model.onnx', test_inputs)
```

## 常见 tf2onnx 标志对照表

| 命令行标志 | Python API 参数 | 说明 |
|-----------|---------------|------|
| `--saved-model DIR` | `saved_model_dir` | SavedModel 目录 |
| `--keras FILE` | N/A (使用 `from_keras`) | Keras H5 文件 |
| `--output FILE` | `output_path` | 输出路径 |
| `--opset N` | `opset` | opset 版本 |
| `--verbose` | `verbose` | 详细日志 |
| `--inputs NODE` | `input_names` | 输入节点名 |
| `--outputs NODE` | `output_names` | 输出节点名 |
| `--extra-opset DOMAIN:VER` | `extra_opset` | 额外 opset |

## 集成到训练流程

```python
import tensorflow as tf
import tf2onnx

def export_model_after_training(model, export_dir, opset=13):
    """训练后自动导出 ONNX"""
    # 1. 保存为 SavedModel
    saved_model_path = f'{export_dir}/saved_model'
    model.save(saved_model_path, save_format='tf')

    # 2. 转换为 ONNX
    onnx_path = f'{export_dir}/model.onnx'
    model_proto, _ = tf2onnx.convert.from_saved_model(
        saved_model_path,
        opset=opset,
        output_path=onnx_path
    )

    # 3. 记录元数据
    import json
    metadata = {
        'tf_version': tf.__version__,
        'opset': opset,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
    with open(f'{export_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ 模型已导出: {onnx_path}")
    return onnx_path

# 在训练脚本中使用
model = tf.keras.applications.EfficientNetB0()
model.compile(optimizer='adam', loss='categorical_crossentropy')
# ... 训练代码 ...
export_model_after_training(model, './exported_model')
```

## 参考资源

- [tf2onnx GitHub](https://github.com/onnx/tensorflow-onnx)
- [tf2onnx 文档](https://github.com/onnx/tensorflow-onnx#usage)
- [[01-基础概念与环境准备/环境搭建指南]] - TensorFlow 环境配置
- [[04-跨框架转换/框架对比分析]] - 框架间转换对比

---

**下一步**: 如需转换其他框架模型，请参考 [[03-流式模型转换/动态轴设置技巧]] 了解动态形状配置，或 [[附录/工具链汇总]] 查看完整的工具生态。
