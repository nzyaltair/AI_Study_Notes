# Linux 与开发环境

## 背景与发展

Linux 是 AI 开发的主流平台，对 GPU 加速、大规模计算和开源框架提供最佳支持。AI 计算集群、云服务器和容器化部署几乎全部基于 Linux。掌握 Linux 命令行与开发环境配置是 AI 工程的基础能力。

## 核心思想

Linux 提供强大的命令行工具链和脚本能力，通过管道（pipe）组合单一功能工具完成复杂任务。AI 开发中利用这一特性实现实验自动化、数据批处理和系统级性能监控，通过 SSH 和容器化保证开发环境与生产环境的一致性。

## 技术原理

### 命令行与管道

核心工具通过管道组合：`grep` 过滤、`wc` 计数、`sed/awk` 文本处理、`sort/uniq` 去重统计。

```bash
# 统计训练日志中 error 出现次数
grep "error" train.log | wc -l
```

### 环境管理

**Conda** 创建隔离的 Python 环境，解决依赖冲突：

```bash
conda create -n ai-env python=3.10
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Docker** 容器化部署确保环境一致性，Dockerfile 定义环境镜像，可在任意机器复现。NVIDIA Docker 扩展支持容器内 GPU 访问。

### GPU 管理

```bash
nvidia-smi                # 查看 GPU 状态与显存占用
nvidia-smi -l 1           # 每秒刷新
CUDA_VISIBLE_DEVICES=0,1 python train.py  # 指定可见 GPU
```

CUDA 环境变量：`CUDA_HOME` 指定 CUDA 安装路径，`LD_LIBRARY_PATH` 包含 CUDA 库。

### Shell 脚本与实验自动化

```bash
#!/bin/bash
# 超参数网格搜索
for lr in 0.001 0.01 0.1; do
    for bs in 32 64 128; do
        python train.py --lr $lr --batch_size $bs \
            --output runs/lr${lr}_bs${bs}
    done
done
```

### 会话管理与远程开发

- **tmux/screen**：SSH 断开后训练进程不中断，可重连查看输出
- **VS Code Remote SSH**：远程编辑与调试
- **rsync**：增量同步代码与数据

### 系统监控

| 层面 | 工具 | 用途 |
|------|------|------|
| CPU/内存 | htop, free, vmstat | 进程资源占用 |
| 磁盘 IO | iotop, du, df | 存储与读写 |
| 网络 | netstat, iperf3 | 连接与带宽 |
| GPU | nvidia-smi, nvtop, gpustat | 显存与计算利用率 |

## 发展演进

物理机部署 → 虚拟化 → Docker 容器化 → Kubernetes 编排 → 云原生 AI 平台（GPU 弹性调度、分布式训练集群管理）。Conda 成为 AI 开发环境管理的事实标准，uv 等新一代工具以更快速度兴起。

## 关键算法·模型

- **管道机制**：Unix 哲学，单一工具组合完成复杂任务
- **容器化**：基于 Linux namespaces 和 cgroups 的进程隔离
- **GPU 调度**：CUDA 驱动层管理 GPU 设备分配与上下文切换

## 应用场景

- 深度学习训练：GPU 集群管理、分布式训练环境配置
- 模型部署：Docker 容器化、服务运维
- 性能监控：系统级瓶颈定位
- 数据工程：大规模数据批处理与传输

## 与其他技术关系

- Linux 是 Python ML 生态的标准运行环境
- GPU 管理与 [[性能优化与调试]] 密切相关
- Docker 容器化支撑 [[软件工程基础]] 中的 CI/CD 实践
- tmux 与 SSH 是远程 AI 开发的基础设施
