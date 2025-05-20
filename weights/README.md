# 模型权重文件

由于模型权重文件较大，未包含在Git仓库中。请将以下模型文件放置在此目录中：

- `v8-impbest.pt` - YOLOv8改进版最佳模型（主要使用）
- `v8-best.pt` - YOLOv8基础版最佳模型（备用）
- `v8-impbest.onnx` - YOLOv8改进版ONNX格式模型（用于推理加速）
- `v8-best.onnx` - YOLOv8基础版ONNX格式模型（备用）
- `frcnn.pth` - Faster-RCNN模型（可选）

## 获取模型文件

您可以通过以下方式获取模型文件：

1. 从原始训练结果中复制
2. 使用模型转换脚本导出ONNX格式
3. 联系项目维护者获取预训练模型 