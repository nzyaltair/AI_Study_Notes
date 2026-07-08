# 容器化与Docker

## 背景与发展

容器化是现代 AI 部署的标准方式。传统部署中，模型依赖（CUDA 版本、Python 库、系统库）的环境不一致导致"开发环境能跑、生产环境报错"的问题频发。2013 年 Docker 提出容器镜像概念，将应用及其依赖打包为可移植的只读镜像。2014 年 Kubernetes 发布，提供容器编排、服务发现和自动扩缩容能力。在 LLM 部署中，容器化将推理引擎、模型权重、CUDA 运行时统一打包，确保跨环境一致性。

## 核心思想

将应用及其全部依赖打包为容器镜像，实现"一次构建，到处运行"。容器相比虚拟机更轻量：共享宿主机内核，启动时间秒级，资源开销低。通过容器编排平台（Kubernetes）管理容器副本的生命周期，实现自动扩缩容、负载均衡和故障恢复。

## 技术原理

### Docker 核心概念

| 概念 | 说明 |
|------|------|
| **镜像（Image）** | 只读模板，包含应用和依赖，通过分层文件系统存储 |
| **容器（Container）** | 镜像的运行实例，具有独立的进程、网络和文件系统空间 |
| **Dockerfile** | 构建镜像的指令文件，每条指令对应一层 |
| **仓库（Registry）** | 镜像存储和分发服务（Docker Hub / Harbor） |
| **卷（Volume）** | 持久化存储，独立于容器生命周期 |
| **网络（Network）** | 容器间通信的虚拟网络 |

### LLM 服务 Dockerfile

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install vllm torch transformers
COPY model/ /app/model/
EXPOSE 8000
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/app/model/", "--port", "8000"]
```

### GPU 容器化

NVIDIA Container Toolkit 使容器内可以访问宿主机 GPU。通过 Docker Compose 声明 GPU 资源：

```yaml
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

### Kubernetes 编排

| K8s 概念 | AI 部署中的作用 |
|---------|----------------|
| **Pod** | 模型推理实例，通常一 Pod 一 GPU |
| **Deployment** | 管理 Pod 副本数和滚动更新 |
| **Service** | 负载均衡和服务发现 |
| **HPA** | 基于 GPU 利用率或 QPS 自动扩缩容 |
| **ConfigMap** | 模型配置和推理参数管理 |
| **PV/PVC** | 模型文件持久化存储 |

### GPU 调度

- **NVIDIA Device Plugin**：K8s 感知和管理 GPU 资源
- **GPU 共享**：时间切片（vGPU）或 MIG（A100/H100 硬件分区）
- **显存管理**：通过 `gpu-memory-utilization` 参数限制单容器显存占用

### 镜像优化

- **多阶段构建**：编译阶段与运行阶段分离，减小最终镜像大小
- **模型分离**：模型文件用 Volume 挂载而非打入镜像，避免镜像膨胀
- **层缓存**：将频繁变更的指令放在 Dockerfile 末尾，利用构建缓存

## 发展演进

虚拟机 → LXC 容器 → Docker → Docker Compose → Kubernetes → GPU 容器化 → K8s GPU Operator → Serverless 容器

## 关键算法·模型

- **UnionFS 分层存储**：镜像通过 OverlayFS 等联合文件系统实现分层叠加，相同层只存储一份，节省磁盘空间和拉取时间
- **cgroups 资源隔离**：限制容器的 CPU、内存、GPU 等资源使用量
- **namespace 命名空间隔离**：为容器提供独立的进程、网络、挂载和用户空间

## 应用场景

- **LLM 服务部署**：vLLM/TGI 容器化部署到 K8s 集群
- **训练任务容器化**：保证训练环境可复现
- **开发环境统一**：DevContainer 提供团队一致的推理开发环境
- **CI/CD 制品**：容器镜像作为部署的标准交付物

## 与其他技术关系

- 容器化是 [[模型服务化]] 的标准部署方式
- [[云端与本地部署]] 依赖容器实现跨环境一致性
- [[监控与日志]] 需要收集容器内应用的指标和日志
- [[版本管理与回滚]] 通过镜像标签管理模型版本
- [[ONNX]] 模型通过容器部署到不同硬件环境
