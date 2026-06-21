# 容器化与Docker

## 1. 简介

容器化技术，特别是Docker，已经成为现代软件部署的标准方式。在大模型部署场景中，容器化技术提供了环境隔离、可复现性、高效部署和资源管理等核心优势，能够有效解决大模型部署中的环境依赖复杂、GPU资源管理、跨平台部署等挑战。

本文档聚焦容器化技术在大模型部署中的应用，涵盖基础原理、工程实践和前沿演进，为AI开发人员和MLOps工程师提供全面的参考。

%% 核心目标：建立大模型容器化部署的最佳实践体系，帮助团队高效、可靠地部署大模型服务 %% ^core-objective

## 2. Docker核心概念

### 2.1 镜像与容器

**镜像（Image）**：
- 静态的、只读的文件系统快照
- 包含运行应用所需的所有依赖（代码、库、环境变量、配置文件等）
- 通过Dockerfile构建，支持分层存储

**容器（Container）**：
- 基于镜像创建的运行实例
- 包含独立的文件系统、网络和进程空间
- 可读写，支持启动、停止、暂停、删除等操作

**关系**：
- 镜像是容器的模板，容器是镜像的运行实例
- 一个镜像可以创建多个容器
- 容器的状态变化不会影响镜像

### 2.2 Dockerfile

**定义**：
- 用于构建Docker镜像的文本文件
- 包含一系列指令，按顺序执行
- 每条指令创建一个新的镜像层

**核心指令**：
- `FROM`：指定基础镜像
- `RUN`：执行命令
- `COPY/ADD`：复制文件到镜像
- `WORKDIR`：设置工作目录
- `ENV`：设置环境变量
- `EXPOSE`：暴露端口
- `CMD/ENTRYPOINT`：容器启动命令

### 2.3 卷（Volumes）

**定义**：
- 用于在容器和主机之间共享或持久化数据
- 绕过容器的可写层，提高性能
- 支持多种驱动（local、nfs、s3等）

**使用场景**：
- 持久化模型权重
- 共享训练数据
- 缓存Hugging Face模型
- 存储日志文件

### 2.4 网络

**网络模式**：
- `bridge`：默认模式，创建隔离网络
- `host`：使用主机网络，无隔离
- `none`：禁用网络
- `container`：共享其他容器的网络

**端口映射**：
- 将容器端口映射到主机端口
- 支持动态端口分配和静态端口映射

### 2.5 多阶段构建

**核心思想**：
- 使用多个FROM指令，每个阶段使用不同的基础镜像
- 只保留最终阶段的文件，减小镜像体积
- 支持构建时和运行时环境分离

**优势**：
- 减小镜像体积
- 提高安全性（不包含构建工具）
- 支持复杂构建流程

## 3. 大模型容器化基础

### 3.1 大模型部署的特殊需求

**GPU资源利用**：
- 需要CUDA支持
- 依赖NVIDIA驱动
- 支持多GPU配置
- GPU内存管理

**大内存占用**：
- 模型权重文件大（7B模型约13GB，70B模型约130GB）
- 推理时需要大量内存
- 支持内存优化技术（如模型量化）

**依赖复杂**：
- 多个Python库依赖
- 版本兼容性要求高
- 系统库依赖（如CUDA、cuDNN）

**高性能要求**：
- 低延迟推理
- 高吞吐量
- 支持批处理

### 3.2 基础镜像选择

**CUDA基础镜像**：
- `nvidia/cuda`：官方NVIDIA CUDA镜像
- 支持多种CUDA版本和Linux发行版
- 提供runtime和devel版本

**Python基础镜像**：
- `python:3.10-slim`：轻量级Python镜像
- `python:3.10-cuda12.1`：预安装CUDA的Python镜像

**框架特定镜像**：
- `pytorch/pytorch`：PyTorch官方镜像
- `tensorflow/tensorflow`：TensorFlow官方镜像
- `huggingface/transformers-pytorch-gpu`：预安装Transformers的镜像

### 3.3 Docker与NVIDIA GPU支持

**NVIDIA Container Toolkit**：
- 允许Docker容器访问NVIDIA GPU
- 提供CUDA驱动和库的容器化支持
- 支持Docker和Kubernetes

**安装与配置**：
```bash
# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**使用GPU容器**：
```bash
docker run --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi
```

## 4. 大模型Dockerfile最佳实践

### 4.1 基础镜像选择

**推荐使用NVIDIA CUDA基础镜像**：
```dockerfile
# 使用CUDA 12.1基础镜像
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 安装Python和依赖管理工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置Python版本
RUN ln -s /usr/bin/python3 /usr/bin/python

# 安装pip依赖
RUN pip install --upgrade pip
```

### 4.2 减小镜像体积

**最佳实践**：
- 使用多阶段构建
- 清理无用文件
- 使用轻量级基础镜像
- 避免安装不必要的依赖
- 使用`--no-install-recommends`选项

**示例**：
```dockerfile
# 构建阶段
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install transformers torch

# 运行阶段
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制依赖
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY app.py .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "app.py"]
```

### 4.3 缓存Hugging Face模型

**问题**：
- 每次构建镜像都会重新下载模型，耗时久
- 模型文件大，增加镜像体积

**解决方案**：
- 使用卷挂载模型目录
- 构建时缓存模型到镜像
- 使用HUGGING_FACE_HUB_CACHE环境变量

**示例**：
```dockerfile
# 设置Hugging Face缓存目录
ENV HUGGING_FACE_HUB_CACHE=/app/model_cache

# 创建缓存目录
RUN mkdir -p $HUGGING_FACE_HUB_CACHE

# 预下载模型到缓存目录
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3-8b-hf'); \
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3-8b-hf');"

# 使用卷挂载模型缓存
VOLUME ["/app/model_cache"]
```

### 4.4 权限管理

**最佳实践**：
- 避免使用root用户运行容器
- 创建专门的用户和组
- 设置适当的文件权限
- 使用Docker的`USER`指令

**示例**：
```dockerfile
# 创建用户和组
RUN groupadd -r llmuser && useradd -r -g llmuser llmuser

# 设置工作目录权限
RUN chown -R llmuser:llmuser /app

# 切换到非root用户
USER llmuser
```

### 4.5 环境变量配置

**推荐配置**：
- 设置CUDA环境变量
- 配置模型路径
- 设置日志级别
- 配置服务端口

**示例**：
```dockerfile
# CUDA环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 模型配置
ENV MODEL_NAME=meta-llama/Llama-3-8b-hf
ENV MODEL_PATH=/app/models

# 服务配置
ENV PORT=8000
ENV HOST=0.0.0.0

# 日志配置
ENV LOG_LEVEL=INFO
```

## 5. 典型部署架构的容器化方案

### 5.1 REST API封装

**架构**：
- 使用FastAPI或Flask构建REST API
- 封装大模型推理逻辑
- 支持同步和异步请求
- 提供OpenAPI文档

**Dockerfile示例**：
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install fastapi uvicorn transformers torch

WORKDIR /app

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**app.py示例**：
```python
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="LLM REST API")

# 加载模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b-hf", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b-hf")

@app.post("/generate")
async def generate(text: str, max_length: int = 100):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
```

### 5.2 gRPC接口

**架构**：
- 使用gRPC构建高性能接口
- 支持双向流和异步通信
- 适合高吞吐量场景
- 强类型定义

**Dockerfile示例**：
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install grpcio grpcio-tools transformers torch

WORKDIR /app

COPY . .

EXPOSE 50051

CMD ["python", "server.py"]
```

### 5.3 异步推理队列

**架构**：
- 使用消息队列（如Redis、RabbitMQ）处理推理请求
- 支持异步推理
- 适合高并发场景
- 支持批处理优化

**Docker Compose示例**：
```yaml
version: '3.8'

services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  llm-worker:
    build: .
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    env_file:
      - .env

  api-server:
    build: .
    command: python api_server.py
    depends_on:
      - redis
    ports:
      - "8000:8000"
    env_file:
      - .env

volumes:
  redis_data:
```

## 6. 完整示例：FastAPI + Transformers的LLM服务容器化

### 6.1 项目结构

```
llm-docker/
├── Dockerfile
├── docker-compose.yml
├── .env
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── models/
│       └── __init__.py
└── requirements.txt
```

### 6.2 requirements.txt

```
fastapi==0.110.0
uvicorn==0.29.0
transformers==4.40.0
torch==2.3.0
accelerate==0.30.0
python-dotenv==1.0.1
```

### 6.3 Dockerfile

```dockerfile
# 使用CUDA 12.1基础镜像
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# 安装pip和依赖
RUN pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY app/ /app/app/

# 创建模型缓存目录
RUN mkdir -p /app/model_cache
ENV HUGGING_FACE_HUB_CACHE=/app/model_cache

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.4 app/main.py

```python
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

app = FastAPI(title="LLM Service", version="1.0.0")

# 配置
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b-hf")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
print(f"Loading model {MODEL_NAME} on {DEVICE}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

@app.get("/")
async def root():
    return {"message": "LLM Service is running!", "model": MODEL_NAME, "device": DEVICE}

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 100, temperature: float = 0.7):
    # 生成文本
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 计算使用的令牌数
    prompt_tokens = len(inputs["input_ids"][0])
    generated_tokens = len(outputs[0]) - prompt_tokens
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": prompt_tokens + generated_tokens
    }
```

### 6.5 docker-compose.yml

```yaml
version: '3.8'

services:
  llm-service:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - model_cache:/app/model_cache
    env_file:
      - .env
    restart: unless-stopped

volumes:
  model_cache:
```

### 6.6 .env文件

```
MODEL_NAME=meta-llama/Llama-3-8b-hf
PORT=8000
HOST=0.0.0.0
```

### 6.7 构建和运行

**构建镜像**：
```bash
docker-compose build
```

**运行服务**：
```bash
docker-compose up -d
```

**测试服务**：
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'
```

## 7. 常见问题排查

### 7.1 CUDA驱动兼容性

**问题**：
- 容器内CUDA版本与主机驱动版本不兼容
- 报错：`CUDA error: invalid device function`

**解决方案**：
- 选择与主机驱动兼容的CUDA版本
- 使用`nvidia-smi`检查主机CUDA驱动版本
- 参考NVIDIA的CUDA兼容性矩阵

**检查命令**：
```bash
# 检查主机CUDA驱动版本
nvidia-smi

# 检查容器内CUDA版本
docker run --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvcc --version
```

### 7.2 权限错误

**问题**：
- 容器内无法访问文件或目录
- 报错：`Permission denied`

**解决方案**：
- 检查文件和目录权限
- 确保容器用户有适当的权限
- 使用`chown`命令设置权限
- 避免使用root用户运行容器

### 7.3 挂载模型缓存目录

**问题**：
- 卷挂载后模型无法访问
- 模型下载到错误的位置

**解决方案**：
- 确保挂载路径正确
- 检查HUGGING_FACE_HUB_CACHE环境变量
- 设置适当的文件权限
- 确认模型文件存在于挂载目录

### 7.4 内存不足

**问题**：
- 容器启动后内存不足
- 报错：`CUDA out of memory`

**解决方案**：
- 增加容器的内存限制
- 使用模型量化技术
- 减小批量大小
- 使用更高效的推理引擎（如vLLM）

**Docker Compose配置**：
```yaml
deploy:
  resources:
    limits:
      memory: 32G
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 8. Docker与新兴部署范式对比

### 8.1 Docker vs WebAssembly

**WebAssembly**：
- 轻量级，启动速度快
- 跨平台，支持多语言
- 沙箱安全模型
- 适合边缘计算和Web应用

**适用边界**：
- Docker：适合复杂的大模型服务，需要完整的操作系统环境
- WebAssembly：适合轻量级推理，边缘设备部署，Web前端推理

### 8.2 Docker vs eBPF容器

**eBPF容器**：
- 基于Linux内核的eBPF技术
- 无需传统容器运行时
- 更低的 overhead
- 更好的资源隔离

**适用边界**：
- Docker：成熟稳定，生态丰富
- eBPF容器：高性能场景，对资源开销敏感的应用

### 8.3 Docker vs Kubernetes Native Serving

**Kubernetes Native Serving**：
- 专为Kubernetes设计的服务框架
- 支持自动扩缩容
- 内置监控和健康检查
- 支持滚动更新和金丝雀发布

**代表项目**：
- Knative Serving
- KFServing
- Seldon Core

**适用边界**：
- Docker：简单部署，开发测试
- Kubernetes Native Serving：大规模生产部署，需要复杂的服务管理功能

## 9. 大模型轻量化部署趋势

### 9.1 ONNX Runtime + Docker

**核心优势**：
- 跨平台支持
- 高性能推理
- 支持模型量化
- 减小镜像体积

**部署示例**：
```dockerfile
FROM mcr.microsoft.com/onnxruntime/server:latest-cuda12-gpu

COPY ./model.onnx /model/model.onnx

ENV MODEL_PATH=/model/model.onnx
ENV MODEL_NAME=llm-model
ENV MODEL_VERSION=1
```

### 9.2 vLLM容器化

**核心优势**：
- 高性能推理
- 动态批处理
- 连续批处理
- 支持多种模型格式

**部署示例**：
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install vllm

EXPOSE 8000

CMD ["vllm", "serve", "meta-llama/Llama-3-8b-hf", "--port", "8000", "--host", "0.0.0.0"]
```

### 9.3 Triton Inference Server集成

**核心优势**：
- 支持多种框架（PyTorch, TensorFlow, ONNX）
- 动态批处理和并行推理
- 内置监控和指标
- 支持GPU和CPU推理

**部署架构**：
```
┌──────────────┐     ┌─────────────────────────┐     ┌──────────────┐
│  Client      │────▶│  Triton Inference Server │────▶│  Model Repository │
└──────────────┘     └─────────────────────────┘     └──────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │  GPU Devices  │
                        └──────────────┘
```

**Docker Compose示例**：
```yaml
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    command: tritonserver --model-repository=/models --http-port=8000 --grpc-port=8001 --metrics-port=8002
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./model_repository:/models
```

## 10. 最佳实践总结

### 10.1 镜像构建最佳实践

1. **选择合适的基础镜像**：使用NVIDIA CUDA基础镜像，根据需求选择runtime或devel版本
2. **使用多阶段构建**：减小镜像体积，提高安全性
3. **缓存模型文件**：使用卷挂载或构建时缓存，避免重复下载
4. **优化依赖管理**：使用requirements.txt，避免不必要的依赖
5. **设置适当的权限**：避免使用root用户，设置正确的文件权限
6. **配置环境变量**：使用ENV指令设置必要的环境变量
7. **添加健康检查**：使用HEALTHCHECK指令监控容器状态

### 10.2 部署最佳实践

1. **使用Docker Compose**：简化多容器部署和管理
2. **配置资源限制**：设置适当的CPU、内存和GPU资源限制
3. **使用卷持久化数据**：特别是模型文件和日志
4. **配置日志管理**：使用集中式日志系统
5. **实现自动扩缩容**：结合Kubernetes或其他编排工具
6. **定期更新镜像**：保持依赖和安全补丁最新
7. **监控容器状态**：使用Prometheus、Grafana等监控工具

### 10.3 性能优化

1. **使用GPU加速**：配置NVIDIA Docker运行时
2. **优化模型加载**：使用预加载和缓存技术
3. **调整批处理大小**：根据硬件资源调整
4. **使用高效的推理引擎**：如vLLM、TensorRT-LLM
5. **启用模型量化**：减小模型大小，提高推理速度
6. **优化内存使用**：使用合适的数据类型，避免内存泄漏

## 11. 关联内容

- 返回 [[../大模型部署与运维|大模型部署与运维]] - 大模型部署与运维主页面
- 参考 [[../模型服务化/模型服务化|模型服务化]] - 模型服务化详细内容
- 参考 [[../vLLM部署/vLLM部署|vLLM部署]] - vLLM部署详细内容
- 参考 [[../../7.大模型推理与优化/GPU加速/GPU加速|GPU加速]] - GPU加速详细内容
- 参考 [[../Kubernetes/Kubernetes|Kubernetes]] - Kubernetes详细内容
- 参考 [[../Docker Compose/Docker Compose|Docker Compose]] - Docker Compose详细内容

## 12. 任务列表

- [ ] 补充多GPU容器配置示例
- [ ] 添加Mermaid流程图，展示容器构建流程
- [ ] 补充Triton Inference Server的完整部署示例
- [ ] 添加WebAssembly推理的Docker部署示例
- [ ] 补充eBPF容器的使用案例
- [ ] 完善vLLM容器化的最佳实践

## 13. 未来展望

### 13.1 容器化技术演进

- **更轻量级的容器运行时**：如containerd、CRI-O
- **更好的GPU支持**：更精细的GPU资源管理
- **更强的安全隔离**：结合硬件辅助虚拟化
- **更高效的镜像格式**：如OCI Image Format v2

### 13.2 大模型部署趋势

- **Serverless LLM**：按需付费，自动扩缩容
- **边缘AI**：将大模型部署到边缘设备
- **模型即服务（MaaS）**：统一的模型服务接口
- **绿色AI**：优化模型推理的能耗

---

*本文档基于2026年1月的最新研究和工业界实践编写，内容将持续更新。*