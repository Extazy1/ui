# 道路目标检测系统

基于YOLOv8的道路目标检测系统，提供了友好的图形用户界面，支持图片、视频和摄像头的实时检测。

## 功能特点

- 支持图片文件检测
- 支持文件夹批量检测
- 支持视频文件检测
- 支持摄像头实时检测
- 可调整检测置信度阈值
- 检测结果分类展示
- 支持检测结果保存

## 环境要求

- Python 3.7+
- PyQt5
- OpenCV
- PyTorch
- Ultralytics YOLO

## 安装方法

```bash
# 克隆仓库
git clone https://github.com/Extazy1/ui.git

# 安装依赖
pip install -r requirements.txt

# 运行程序
python ui.py
```

## 使用说明

1. 启动程序后，系统将自动加载默认模型
2. 选择输入源（图片、文件夹、视频或摄像头）
3. 调整检测参数（如需要）
4. 查看检测结果并保存 