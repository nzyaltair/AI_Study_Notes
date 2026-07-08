# ONNX 模型导出

[[ONNX]]（Open Neural Network Exchange）是 Facebook/Microsoft 于 2017 年推出的开放模型交换格式，解决 PyTorch、TensorFlow 等框架间的互操作问题。模型导出为 ONNX 后，可在 [[ONNX Runtime]]、[[TensorRT]]、OpenVINO 等推理引擎上跨平台部署。

## 子主题导航

| 编号 | 文件 | 内容 |
|------|------|------|
| 01 | [[01_ONNX格式与基础]] | ONNX IR、Opset 版本、序列化格式、环境搭建 |
| 02 | [[02_模型转换方法]] | torch.onnx.export 参数详解、TF2ONNX、PaddlePaddle 转换 |
| 03 | [[03_流式与动态模型转换]] | 动态轴设置、KV Cache、编码器/解码器分离 |
| 04 | [[04_跨框架部署与优化]] | ONNX→TensorRT、ONNX Runtime、框架对比、算子兼容性 |
| 05 | [[05_常见问题与算子兼容]] | 内存优化、动态维度、数据类型匹配、自定义算子 |
| 06 | [[06_验证评估与场景优化]] | 正确性验证、精度调优、性能测试、云端/边缘部署 |

## 学习路径

- **入门**：01 → 02（PyTorch 转换）→ 06（验证）
- **进阶**：03（流式模型）→ 04（TensorRT 优化）→ 05（问题排查）
- **生产部署**：04（跨框架选择）→ 06（云端/边缘场景优化）

## 核心工具链

- `torch.onnx.export` — PyTorch 内置导出
- `tf2onnx` — TensorFlow 转 ONNX
- `paddle2onnx` — PaddlePaddle 转 ONNX
- [[ONNX Runtime]] — 跨平台推理引擎
- [[TensorRT]] — NVIDIA GPU 极致优化
- Netron — 模型可视化
- onnx-simplifier — 图结构简化

## 相关链接

- [[模型部署]] — 部署流程总览
- [[模型推理]] — 推理引擎对比
- [[量化]] — INT8/FP16 量化技术
- ONNX 官方文档：https://onnx.ai/
- ONNX Runtime 文档：https://onnxruntime.ai/
