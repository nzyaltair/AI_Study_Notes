# Linux与开发环境

## 背景

Linux是AI开发的主流平台，对GPU加速、大规模计算和开源AI框架有最佳支持。大多数AI计算集群和云服务器基于Linux系统。掌握Linux命令行和开发环境配置是AI工程师的基本技能。

## 核心思想

Linux提供强大的命令行工具和脚本能力，便于自动化实验管理、大规模数据处理和系统级性能调优。

## 技术原理

### 常用命令

- **文件操作**：`ls`、`cd`、`cp`、`mv`、`rm`、`find`、`tree`
- **文本处理**：`cat`、`grep`、`sed`、`awk`、`head`、`tail`、`wc`
- **管道与重定向**：`|`、`>`、`>>`、`2>`
  ```bash
  # 查找包含error的日志行并统计
  grep "error" train.log | wc -l
  ```
- **进程管理**：`ps`、`top`、`htop`、`kill`、`nvidia-smi`
- **SSH远程**：`ssh user@host`、`scp`、`rsync`

### 环境管理

- **Conda**：创建隔离的Python环境
  ```bash
  conda create -n ai-env python=3.10
  conda activate ai-env
  conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
  ```
- **venv**：Python原生虚拟环境
- **Docker**：容器化部署，确保环境一致性

### GPU管理

- **nvidia-smi**：监控GPU使用情况
  ```bash
  nvidia-smi  # 查看GPU状态
  nvidia-smi -l 1  # 每秒刷新
  ```
- **CUDA环境**：`CUDA_HOME`、`LD_LIBRARY_PATH`
- **多GPU使用**：`CUDA_VISIBLE_DEVICES=0,1 python train.py`

### Shell脚本

```bash
#!/bin/bash
# 批量训练脚本
for lr in 0.001 0.01 0.1; do
    for batch_size in 32 64 128; do
        echo "Training with lr=$lr batch_size=$batch_size"
        python train.py --lr $lr --batch_size $batch_size
    done
done
```

### 开发工具

- **编辑器**：VS Code（Remote SSH）、Vim/Neovim
- **终端工具**：tmux（会话管理）、screen
- **文件传输**：scp、rsync、rclone（云存储）
- **实验管理**：Weights & Biases、MLflow、TensorBoard

### 系统监控

- **CPU/内存**：`top`、`htop`、`free -h`、`vmstat`
- **磁盘**：`df -h`、`du -sh *`、`iotop`
- **网络**：`netstat`、`ss`、`iperf3`
- **GPU**：`nvidia-smi`、`gpustat`、`nvtop`

### 定时任务

```bash
# crontab定时执行
crontab -e
# 每天凌晨3点训练模型
0 3 * * * cd /project && python train.py >> logs/train.log 2>&1
```

## 应用领域

- **[[深度学习]]训练**：GPU集群管理、实验自动化
- **[[系统工程化与运维]]**：模型部署、服务运维
- **[[性能优化与调试]]**：系统级性能监控
- **[[数据处理与工程化|数据工程]]**：大规模数据处理

## 与其他技术关系

- Linux是[[Python基础|Python开发]]的标准运行环境
- Linux GPU管理与[[性能优化与调试|GPU优化]]密切相关
- Linux部署与[[系统工程化与运维|Docker容器化]]配合使用
