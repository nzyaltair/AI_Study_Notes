# 容器化与Docker

## 背景

容器化是现代AI部署的标准方式，通过Docker将模型、依赖和运行环境打包为可移植的镜像，确保开发、测试和生产环境的一致性。Kubernetes提供了容器编排能力，支持大规模模型部署和运维。

## 核心思想

将应用及其依赖打包为容器镜像，实现"一次构建，到处运行"。通过容器编排平台实现自动扩缩容、负载均衡和故障恢复。

## 技术原理

### Docker基础

```dockerfile
# LLM服务Dockerfile示例
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install vllm torch transformers

COPY model/ /app/model/
COPY server.py /app/

EXPOSE 8000
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/app/model/", "--port", "8000"]
```

### Docker核心概念

| 概念 | 说明 |
|------|------|
| **镜像（Image）** | 只读模板，包含应用和依赖 |
| **容器（Container）** | 镜像的运行实例 |
| **Dockerfile** | 构建镜像的指令文件 |
| **仓库（Registry）** | 镜像存储和分发（Docker Hub/Harbor） |
| **卷（Volume）** | 持久化存储 |
| **网络（Network）** | 容器间通信 |

### GPU容器化

```yaml
# docker-compose.yml with GPU
services:
  llm-server:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./model:/model
    command: --model /model --port 8000
```

### Kubernetes编排

| K8s概念 | 在AI部署中的作用 |
|---------|----------------|
| **Pod** | 模型推理实例 |
| **Deployment** | 管理Pod副本和更新 |
| **Service** | 负载均衡和服务发现 |
| **HPA** | 基于负载自动扩缩容 |
| **ConfigMap** | 模型配置管理 |
| **PV/PVC** | 模型文件持久化 |

### GPU调度

- **NVIDIA Device Plugin**：K8s GPU资源管理
- **GPU共享**：时间切片/MIG（Multi-Instance GPU）
- **显存管理**：限制单容器GPU显存使用
- **队列调度**：多团队共享GPU集群

### 镜像优化

- **多阶段构建**：减小镜像大小
- **基础镜像选择**：CUDA基础镜像 vs 精简镜像
- **层缓存**：合理排序Dockerfile指令
- **模型分离**：模型文件用Volume挂载，不打入镜像

## 发展演进

虚拟机 → Docker容器 → Docker Compose → Kubernetes → Serverless容器 → GPU容器化 → K8s GPU Operator

## 应用领域

- **[[模型部署与服务化|LLM服务部署]]**：vLLM/TGI容器化
- **[[机器学习流水线|ML Pipeline]]**：训练任务容器化
- **[[Agent智能体|Agent应用]]**：Agent服务容器化
- **开发环境**：统一的开发容器（DevContainer）
- **CI/CD**：容器镜像作为部署制品

## 与其他技术关系

- 容器化是[[模型部署与服务化|模型部署]]的标准方式
- [[监控与运维|监控]]需要收集容器指标
- [[软件工程基础|Docker]]是[[编程与工程基础|编程工程]]的核心工具
- K8s与[[机器学习流水线|MLOps]]平台集成
- [[ONNX模型导出|ONNX]]模型通过容器部署到不同环境
