# ONNX到TensorRT优化

## 概述

TensorRT是NVIDIA的高性能推理优化器和运行时，能够将ONNX模型转换为高度优化的TensorRT引擎。本文档详细介绍转换流程、参数调优和最佳实践。

## 转换流程概览

```mermaid
graph LR
    A[ONNX模型] --> B{分析模型}
    B --> C[层融合]
    C --> D[精度校准]
    D --> E[内核自动调整]
    E --> F[TensorRT引擎]
    F --> G[部署推理]
```

## TensorRT安装配置

### CUDA环境要求

| TensorRT版本 | CUDA版本 | cuDNN版本 | Python版本 |
|--------------|---------|----------|-----------|
| 10.x | 12.x | 8.x | 3.8-3.11 |
| 9.x | 11.8 | 8.x | 3.8-3.10 |
| 8.x | 11.6 | 8.x | 3.6-3.10 |

### 安装步骤

```bash
# 方式1: pip安装（推荐）
pip install tensorrt

# 方式2: NVIDIA apt仓库（Ubuntu）
sudo apt-get install tensorrt

# 验证安装
python -c "import tensorrt as trt; print(trt.__version__)"
```

## trtexec命令行工具深度解析

### 基础转换命令

```bash
#  simplest转换
trtexec --onnx=model.onnx --saveEngine=model.engine

# 完整参数示例
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine \
    --fp16 \
    --workspace=4096 \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:8x3x224x224 \
    --maxShapes=input:32x3x224x224 \
    --verbose \
    --timingCacheFile=timing.cache
```

### 关键参数详解

#### 1. 精度模式 (`--fp16`, `--int8`, `--best`)

| 参数 | 精度 | 内存占用 | 推理速度 | 精度损失 |
|------|------|---------|---------|---------|
| 默认 | FP32 | 1.0x | 1.0x | 0% |
| `--fp16` | FP16 | ~0.6x | 1.5-2x | <1% |
| `--int8` | INT8 | ~0.5x | 2-4x | 1-2% |
| `--best` | 自动选择 | 最低 | 最快 | 最小 |

**使用示例：**

```bash
# FP16加速（无需校准）
trtexec --onnx=model.onnx --fp16 --saveEngine=model_fp16.engine

# INT8量化（需要校准）
trtexec --onnx=model.onnx --int8 \
        --calib=calibration.cache \
        --saveEngine=model_int8.engine
```

#### 2. Workspace设置 (`--workspace`)

- **定义**: GPU上可用于优化的临时内存（MB）
- **推荐值**:
  - CNN模型: 2048-4096 MB
  - Transformer模型: 4096-8192 MB
  - 超大模型: 10240+ MB
- **影响**: 过小会导致转换失败；过大影响多模型并发

**最佳实践：**
```bash
# 根据GPU内存动态调整
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
WORKSPACE=$((GPU_MEM / 4))  # 使用25% GPU内存
trtexec --onnx=model.onnx --workspace=$WORKSPACE ...
```

#### 3. 动态形状配置 (`--minShapes`, `--optShapes`, `--maxShapes`)

**格式**: `name:dim0xdim1x...`

```bash
# 单输入例子（batch size动态）
--minShapes=input:1x3x224x224 \
--optShapes=input:8x3x224x224 \
--maxShapes=input:32x3x224x224

# 多输入例子
--minShapes=input1:1x3x224x224,input2:1x10 \
--optShapes=input1:8x3x224x224,input2:8x10 \
--maxShapes=input1:32x3x224x224,input2:32x10
```

**动态维度规则：**
- 所有维度必须满足：`min[i] ≤ opt[i] ≤ max[i]`
- 通常以 batch size 为动态维度
- 固定维度使用相同值（如：`224`）

#### 4. 性能调优参数

```bash
# 层融合策略
trtexec --onnx=model.onnx \
        --heuristic=aggressive \
        --layerPrecisions=conv:fp16,fc:fp16

# Timing缓存加速构建
trtexec --onnx=model.onnx \
        --timingCacheFile=timing.cache \
        --forceTimingCache=1

# 多线程优化
trtexec --onnx=model.onnx \
        --numThreads=$(nproc)
```

### 完整选项清单

| 类别 | 参数 | 说明 | 常用值 |
|------|------|------|-------|
| **输入输出** | `--onnx` | ONNX文件路径 | required |
| | `--saveEngine` | 引擎保存路径 | `model.engine` |
| | `--inputs` | 输入名称列表 | `input` |
| | `--outputs` | 输出名称列表 | `output` |
| **精度** | `--fp16` | 启用FP16 | - |
| | `--int8` | 启用INT8 | - |
| | `--noTF32` | 禁用TF32 | - |
| **内存** | `--workspace` | 工作空间大小(MB) | 4096 |
| | `--maxBatch` | 最大batch（旧版本） | 32 |
| **形状** | `--minShapes` | 最小形状 | `input:1x...` |
| | `--optShapes` | 最优形状 | `input:8x...` |
| | `--maxShapes` | 最大形状 | `input:32x...` |
| **校准** | `--calib` | 校准缓存文件 | `calib.cache` |
| | `--calibration` | 校准数据集 | `./calib_data/` |
| | `--calibration_batch_size` | 校准batch size | 8 |
| **优化** | `--heuristic` | 启发式融合策略 | `aggressive` |
| | `--layerPrecisions` | 层精度映射 | `conv:fp16` |
| | `--timingCacheFile` |  timing缓存 | `timing.cache` |
| **调试** | `--verbose` | 详细输出 | - |
| | `--dumpLayerinfo` | 导出层信息 | - |
| | `--exportProfile` | 导出性能分析 | `profile.json` |
| **其他** | `--device` | 指定GPU | `cuda:0` |
| | `--numThreads` | 工作线程数 | CPU核心数 |

## INT8量化校准详解

### 校准方法对比

| 方法 | 原理 | 数据量 | 精度影响 | 速度 |
|------|------|-------|---------|------|
| **EntropyCalib2** | 熵最大化 | 500-1000张 | 最小 | 快 |
| **MinMaxCalib** | 最小-最大 | 100-500张 | 较小 | 快 |
| **LegacyCalib** | 旧版算法 | 1000+张 | 较大 | 慢 |

**推荐**: 使用 `EntropyCalib2`（TensorRT默认）

### 校准数据集准备

```python
import numpy as np
import tensorrt as trt

# 校准数据生成器
class CalibrationDataGenerator(trt.IInt8Calibrator):
    def __init__(self, data_path, batch_size, input_shape):
        trt.IInt8Calibrator.__init__(self)
        self.data = self.load_calibration_data(data_path)
        self.batch_size = batch_size
        self.index = 0

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.data):
            return None  # 校准完成

        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size

        # 返回batch数据（作为list of numpy arrays）
        return [batch]

    def read_calibration_cache(self, cache_file):
        # 读取已有缓存（加速重复校准）
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open("calibration.cache", "wb") as f:
            f.write(cache)
```

### Python API校准示例

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 1. 创建Builder和Network
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

# 2. 解析ONNX
parser = trt.OnnxParser(network, logger)
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# 3. 配置
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 << 20)

# 4. 设置INT8校准
config.set_flag(trt.BuilderFlag.INT8)
calibrator = CalibrationDataGenerator(
    calibration_data_path="./calib_dataset/",
    batch_size=8,
    input_shape=(1, 3, 224, 224)
)
config.set_calibration_profile(profile)
config.set_calibrator(calibrator)

# 5. 构建引擎
engine = builder.build_engine(network, config)

# 6. 保存引擎
with open("model_int8.engine", "wb") as f:
    f.write(engine.serialize())
```

## 完整示例：BERT到TensorRT（含INT8校准）

### 步骤1: ONNX导出（PyTorch）

```python
import torch
from transformers import BertModel

# 加载预训练BERT
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# 输入示例
input_ids = torch.ones(1, 128, dtype=torch.long)
attention_mask = torch.ones(1, 128, dtype=torch.long)

# 导出ONNX
torch.onnx.export(
    model,
    (input_ids, attention_mask, None, None),
    "bert.onnx",
    opset_version=14,
    input_names=['input_ids', 'attention_mask', 'token_type_ids', 'position_ids'],
    output_names=['last_hidden_state', 'pooler_output'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'seq_len'},
        'attention_mask': {0: 'batch', 1: 'seq_len'}
    },
    do_constant_folding=True
)
```

### 步骤2: 校准数据准备

```python
import numpy as np
from datasets import load_dataset

# 使用HuggingFace数据集
dataset = load_dataset("glue", "mrpc", split="validation")

def preprocess(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

calibration_data = []
for i in range(100):  # 100个样本用于校准
    sample = dataset[i]
    inputs = tokenizer(
        sample["sentence1"],
        sample["sentence2"],
        return_tensors="pt"
    )
    calibration_data.append({
        'input_ids': inputs['input_ids'].numpy(),
        'attention_mask': inputs['attention_mask'].numpy()
    })

np.save('calib_data.npy', calibration_data)
```

### 步骤3: TensorRT转换（trtexec）

```bash
# FP16推理引擎（快速）
trtexec \
    --onnx=bert.onnx \
    --saveEngine=bert_fp16.engine \
    --fp16 \
    --workspace=8192 \
    --minShapes=input_ids:1x128,attention_mask:1x128 \
    --optShapes=input_ids:8x128,attention_mask:8x128 \
    --maxShapes=input_ids:32x128,attention_mask:32x128 \
    --verbose

# INT8校准引擎（最高性能）
trtexec \
    --onnx=bert.onnx \
    --saveEngine=bert_int8.engine \
    --int8 \
    --calibration=./calib_dataset \
    --workspace=8192 \
    --minShapes=input_ids:1x128,attention_mask:1x128 \
    --optShapes=input_ids:8x128,attention_mask:8x128 \
    --maxShapes=input_ids:32x128,attention_mask:32x128
```

### 步骤4: Python推理测试

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, input_dict):
        # 分配设备内存
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        # 拷贝输入数据
        np.copyto(inputs[0]['host'], input_dict['input_ids'])
        np.copyto(inputs[1]['host'], input_dict['attention_mask'])

        # 推理
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        cuda.memcpy_htod_async(inputs[1]['device'], inputs[1]['host'], stream)
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

        return outputs[0]['host'].reshape(1, 128, 768)
```

## 性能调优最佳实践

### 1. Workspace优化策略

```bash
# 策略A: 保守模式（稳定优先）
--workspace=2048

# 策略B: 平衡模式（推荐）
--workspace=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) * 4096 / 16384 ))

# 策略C: 激进模式（性能优先）
--workspace=16384  # 注意：需确保GPU内存足够
```

### 2. 多精度混合策略

```bash
# 对特定层使用不同精度
--layerPrecisions=conv:fp16,fc:fp16,layernorm:fp32

# 结合FP16和INT8（高级）
trtexec --onnx=model.onnx \
        --fp16 \
        --int8 \
        --calib=calib.cache \
        --layerPrecisions=fp16:conv1,conv2,int8:fc1,fc2
```

### 3. 动态形状优化

```bash
# 针对特定工作负载优化
# 场景1: 推理服务（小batch）
--minShapes=input:1x3x224x224 \
--optShapes=input:4x3x224x224 \
--maxShapes=input:16x3x224x224

# 场景2: 批量处理（大batch）
--minShapes=input:8x3x224x224 \
--optShapes=input:64x3x224x224 \
--maxShapes=input:128x3x224x224
```

### 4. Timing Cache利用

```bash
# 第一次构建时生成缓存
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --workspace=4096 \
        --timingCacheFile=timing.cache

# 第二次构建复用缓存（加速50%+）
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --workspace=4096 \
        --timingCacheFile=timing.cache \
        --forceTimingCache=1
```

### 5. 层融合控制

```bash
# 激进融合（尝试最大优化）
trtexec --onnx=model.onnx \
        --heuristic=aggressive \
        --verbose

# 保守融合（保证稳定性）
trtexec --onnx=model.onnx \
        --heuristic=conservative \
        --skipLayerNormToMatrixMultiplication=0

# 禁用特定融合
trtexec --onnx=model.onnx \
        --disableConvolutionLayerFusion=1 \
        --disableMatrixMultiplicationLayerFusion=1
```

## 常见问题排查

### 问题1: Workspace不足

```
[ERROR] Out of memory: workspace size is not enough
```

**解决方案：**
```bash
# 增加workspace
trtexec --workspace=8192 ...

# 或启用layer precision减少内存
trtexec --fp16 ...
```

### 问题2: 不支持的operator

```
[ERROR] Unsupported operator: MyCustomOp
```

**解决方案：**
1. 检查ONNX模型是否包含自定义op
2. 使用 `onnxruntime-extensions` 扩展支持
3. 或修改原模型避免该op

```bash
# 查看所有层
trtexec --onnx=model.onnx --dumpLayerInfo
```

### 问题3: INT8校准失败

```
[ERROR] Calibration failed: No valid data
```

**解决方案：**
```python
# 确保校准数据正确归一化（匹配训练时）
calibration_data = preprocess(data) / 255.0  # 例如图像

# 检查数据分布
print(f"Calib data shape: {calibration_data.shape}")
print(f"Calib data range: [{calibration_data.min()}, {calibration_data.max()}]")
```

### 问题4: 动态形状配置错误

```
[ERROR] Invalid dynamic range: max < min
```

**解决方案：**
```bash
# 确保: min ≤ opt ≤ max
--minShapes=input:1x224x224 \
--optShapes=input:8x224x224 \
--maxShapes=input:32x224x224  # ✓ correct

--minShapes=input:32x224x224 \
--optShapes=input:8x224x224 \
--maxShapes=input:1x224x224  # ✗ wrong
```

## 性能分析

### 使用--exportProfile

```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --exportProfile=profile.json \
        --verbose

# 分析profile（使用TensorRT Profiler或nsys）
nsys profile -o trt_profile python inference.py
```

### 关键指标

| 指标 | 说明 | 优化目标 |
|------|------|---------|
| **Enqueue Time** | 数据拷贝+内核启动 | < 1ms |
| **H2D Time** | Host到Device拷贝 | < 0.5ms |
| **D2H Time** | Device到Host拷贝 | < 0.5ms |
| **Kernel Time** | 实际计算时间 | 越小越好 |
| **Layer Time** | 每层执行时间 | 识别瓶颈 |

## 部署建议

### 1. 生产环境引擎构建

```bash
#!/bin/bash
# build_engine.sh - 生产级引擎构建脚本

set -e

MODEL_PATH=$1
ENGINE_PATH=$2
PRECISION=${3:-fp16}
WORKSPACE=${4:-4096}

echo "Building TensorRT engine..."
echo "  Model: $MODEL_PATH"
echo "  Output: $ENGINE_PATH"
echo "  Precision: $PRECISION"
echo "  Workspace: ${WORKSPACE}MB"

if [ "$PRECISION" == "int8" ]; then
    trtexec \
        --onnx=$MODEL_PATH \
        --saveEngine=$ENGINE_PATH \
        --int8 \
        --calib=calib.cache \
        --workspace=$WORKSPACE \
        --verbose \
        2>&1 | tee build.log
else
    trtexec \
        --onnx=$MODEL_PATH \
        --saveEngine=$ENGINE_PATH \
        --$PRECISION \
        --workspace=$WORKSPACE \
        --verbose \
        2>&1 | tee build.log
fi

if [ $? -eq 0 ]; then
    echo "✅ Engine built successfully!"
    ls -lh $ENGINE_PATH
else
    echo "❌ Engine build failed!"
    exit 1
fi
```

### 2. Docker化构建

```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/tensorrt:23.10-py3

COPY model.onnx /app/
COPY build_engine.sh /app/
COPY calib_dataset /app/calib_dataset/

WORKDIR /app
RUN pip install torch transformers datasets

CMD ["./build_engine.sh", "model.onnx", "model.engine", "fp16"]
```

### 3. 版本管理

```bash
# 记录构建环境
cat > build_info.txt << EOF
TensorRT Version: $(trtexec --version | head -1)
CUDA Version: $(nvcc --version | grep "release" | awk '{print $5}')
ONNX Model: model.onnx (md5: $(md5sum model.onnx | cut -d' ' -f1))
Build Date: $(date)
Build Params: $@
EOF
```

## 与ONNX Runtime对比

| 特性 | TensorRT | ONNX Runtime | 选择建议 |
|------|---------|-------------|---------|
| **推理速度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 追求极致性能选TRT |
| **精度控制** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 需要INT8选TRT |
| **硬件支持** | NVIDIA GPU only | CPU/GPU/多Vendor | 多平台选ORT |
| **API复杂度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 快速上手选ORT |
| **模型兼容性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 复杂模型选ORT |
| **内存占用** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 内存紧张取决于场景 |

**通用策略：**
- NVIDIA GPU服务器：优先TensorRT
- CPU或多GPU混合环境：ONNX Runtime
- 原型验证：ONNX Runtime（快速迭代）
- 生产部署：两者都测试，选更优者

## 参考资源

- [[07-场景优化建议/硬件平台适配]]
- TensorRT官方文档：https://docs.nvidia.com/deeplearning/tensorrt/
- NVIDIA Developer Blog：TensorRT最佳实践
- ONNX GitHub：https://github.com/onnx/onnx

---

**标签**: #tensorrt #nvidia #gpu-optimization
