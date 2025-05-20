import os
from pathlib import Path
from ultralytics import YOLO

def predict_simple(model_path, source_dir, output_dir):
    """
    精简版预测函数
    model_path: 模型路径(.pt)
    source_dir: 输入图像目录
    output_dir: 输出目录
    """
    # 加载模型 [1,4](@ref)
    model = YOLO(model_path)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 批量预测并自动保存 [1,6](@ref)
    results = model.predict(
        source=source_dir, 
        save=True,  # 自动保存带标注的图片
        project=output_dir,  # 指定输出目录
        exist_ok=True,  # 允许覆盖已有文件
        show_conf=False  # 隐藏置信度分数
    )

if __name__ == "__main__":
    # 直接配置参数
    config = {
        "model_path": "./weights/v8-impbest.pt",  # 默认模型路径
        "source_dir": "./bdd",  # 输入图像目录
        "output_dir": "./bdd_predict"  # 输出目录
    }
    
    # 执行预测
    predict_simple(**config)