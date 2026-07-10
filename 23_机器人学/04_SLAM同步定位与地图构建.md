# SLAM同步定位与地图构建

## 一句话理解

机器人在未知环境中同时估计自身位姿（定位）和构建环境地图（建图）——是自主移动机器人的核心技术。

## 背景与发展

SLAM是机器人学中最重要的问题之一。从Smith & Cheeseman的EKF-SLAM到Montemerlo的FastSLAM，再到Mur-Artal的ORB-SLAM，SLAM技术经历了从滤波到图优化、从激光到视觉的演进。

| 年代 | 里程碑 | 作者 | 意义 |
|:---|:---|:---|:---|
| 1986 | SLAM问题提出 | Smith & Cheeseman | 统一定位与建图 |
| 1989 | EKF-SLAM | Smith等 | 扩展卡尔曼滤波SLAM |
| 2002 | FastSLAM | Montemerlo等 | 粒子滤波SLAM |
| 2006 | Graph-SLAM | Thrun等 | 图优化方法 |
| 2007 | GMapping | Grisetti等 | 开源激光SLAM |
| 2015 | ORB-SLAM | Mur-Artal等 | 特征点视觉SLAM |
| 2016 | LSD-SLAM | Engel等 | 直接法视觉SLAM |
| 2017 | ORB-SLAM2 | Mur-Artal等 | 支持双目/RGB-D |
| 2020 | ORB-SLAM3 | Campos等 | 支持IMU多传感器融合 |
| 2020s | 神经辐射场SLAM | 多家 | NeRF/3DGS + SLAM |

## 核心概念

### SLAM问题的本质

SLAM的核心困难在于**鸡生蛋问题**：
- 精确定位需要准确的地图
- 准确建图需要精确的定位

两者必须同时估计，形成联合优化问题：

$$p(\mathbf{x}_{0:t}, \mathbf{m} | \mathbf{z}_{1:t}, \mathbf{u}_{1:t})$$

- $\mathbf{x}_{0:t}$：机器人轨迹（位姿序列）
- $\mathbf{m}$：地图
- $\mathbf{z}_{1:t}$：观测序列
- $\mathbf{u}_{1:t}$：控制输入

### SLAM方法分类

| 维度 | 分类 | 说明 |
|:---|:---|:---|
| 传感器 | 激光SLAM / 视觉SLAM | 激光精度高，视觉信息丰富 |
| 估计方法 | 滤波式 / 图优化 | 滤波在线更新，图优化批量优化 |
| 地图表示 | 栅格地图 / 特征地图 / 点云地图 | 不同表示适合不同场景 |

### 前端 vs 后端

SLAM系统通常分为前端和后端：

```
传感器数据 → 前端（位姿跟踪/数据关联） → 后端（优化） → 地图
                    ↑                              ↓
                    └──── 回环检测 ────────────────┘
```

- **前端**：帧间运动估计、特征匹配、数据关联
- **后端**：位姿图优化、全局一致性
- **回环检测**：识别已访问过的地点，消除累积漂移

## 工作原理

### EKF-SLAM

使用扩展卡尔曼滤波同时估计机器人位姿和路标位置。

状态向量：$\mathbf{x} = [\mathbf{x}_r, \mathbf{m}_1, \ldots, \mathbf{m}_n]^T$

预测步骤：
$$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k)$$
$$P_{k|k-1} = F_x P_{k-1} F_x^T + F_u Q F_u^T$$

更新步骤：
$$K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}$$
$$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_{k|k-1} + K_k(\mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1}))$$

缺点：状态维度随路标数线性增长，计算复杂度 $O(n^2)$，不适合大规模环境。

### FastSLAM

基于粒子滤波，利用"机器人轨迹和地图在给定轨迹条件下独立"的性质分解问题：

$$p(\mathbf{x}_{0:t}, \mathbf{m} | \mathbf{z}_{1:t}, \mathbf{u}_{1:t}) = p(\mathbf{x}_{0:t} | \mathbf{z}_{1:t}, \mathbf{u}_{1:t}) \prod_i p(\mathbf{m}_i | \mathbf{x}_{0:t}, \mathbf{z}_{1:t})$$

每个粒子维护一条轨迹假设和对应的EKF路标地图。FastSLAM将计算复杂度降至 $O(M \log n)$（$M$ 为粒子数）。

### Graph-SLAM

将SLAM建模为**因子图优化**问题：

$$\min_{\mathbf{x}, \mathbf{m}} \sum_t \|\mathbf{z}_t - h(\mathbf{x}_t, \mathbf{m}_t)\|^2_{\Sigma_t^{-1}} + \sum_t \|\mathbf{u}_t - g(\mathbf{x}_{t-1}, \mathbf{x}_t)\|^2_{Q_t^{-1}}$$

构建位姿图后使用Gauss-Newton或Levenberg-Marquardt优化。回环检测加入闭环约束后，图优化能消除累积漂移。

### ORB-SLAM

最完整的视觉SLAM系统，包含三个线程：

1. **跟踪**（Tracking）：实时估计相机位姿
2. **局部建图**（Local Mapping）：管理局部地图点，局部BA优化
3. **回环检测**（Loop Closing）：DBoW2词袋检测回环，全局BA

使用ORB特征点，支持单目/双目/RGB-D，是学术界的标杆系统。

### LIO（LiDAR-Inertial Odometry）

激光+IMU融合的里程计方法（如LIO-SAM），通过紧耦合融合实现高精度定位。

## 应用场景

| 场景 | SLAM类型 |
|:---|:---|
| 室内服务机器人 | 激光SLAM（GMapping/Cartographer） |
| 仓储AGV | 激光SLAM |
| 自动驾驶 | LiDAR-Visual-Inertial SLAM |
| 无人机 | 视觉SLAM/VIO |
| AR/VR | 视觉SLAM（VINS/ORB-SLAM3） |
| 室外测绘 | 激光SLAM + GPS |

## 优势与不足

### 优势
- 实现无GPS环境下的自主定位
- 图优化方法精度高、全局一致
- 视觉SLAM信息丰富，可支持语义建图

### 不足
- 动态环境影响鲁棒性
- 弱纹理/弱结构场景视觉SLAM失效
- 大规模环境计算开销大
- 长期运行的地图维护和更新困难

## 相关知识

- [[02_传感器与感知]] — 传感器融合是SLAM的基础
- [[01_数学基础]] — 概率论、线性代数和优化是SLAM的数学基础
- [[15_计算机视觉]] — 视觉SLAM使用特征提取、光流等CV技术
- [[13_强化学习]] — SLAM为机器人导航提供环境模型
- [[00_机器人学_综述]] — 本方向综述

## References

- Thrun, Burgard & Fox, *Probabilistic Robotics* (2005)
- Mur-Artal & Tardós, *ORB-SLAM2* (IEEE T-RO, 2017)
- Cadena et al., *Past, Present, and Future of Simultaneous Localization and Mapping* (IEEE T-RO, 2016)
- Grisetti et al., *A Tutorial on Graph-Based SLAM* (IEEE Intelligent Transportation Systems Magazine, 2010)
