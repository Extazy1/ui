import os
import sys
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 设置测试数据集路径
    dataset_path = r"E:\Develop\object-dection\DataSets\final_dataset_5_cleaned"
    
    # 设置测试模型路径 (默认使用最好的模型)
    model_path = r"weights/v8-impbest.pt"
    
    # 确认路径是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        sys.exit(1)
    
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 构建测试数据集的yaml文件路径
    dataset_yaml = os.path.join(dataset_path, "bdd.yaml")
    
    # 运行测试评估
    print("开始评估模型性能...")
    results = model.val(
        data=dataset_yaml,
        split="val",  # 使用验证集进行测试
        verbose=True,
        save_json=True,
        save_conf=True,
        plots=True
    )
    
    # 打印测试结果指标
    print("\n模型评估指标:")
    print(f"mAP@0.5: {results.box.map50:.4f}")  # mAP@0.5
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")  # mAP@0.5:0.95
    print(f"精确率 (Precision): {results.box.p:.4f}")
    print(f"召回率 (Recall): {results.box.r:.4f}")
    print(f"F1-Score: {2 * (results.box.p * results.box.r) / (results.box.p + results.box.r + 1e-16):.4f}")
    
    # 如果需要，可以保存结果或生成图表
    result_path = "test_results"
    os.makedirs(result_path, exist_ok=True)
    
    # 将结果保存到文本文件
    with open(os.path.join(result_path, "metrics.txt"), "w") as f:
        f.write(f"模型: {model_path}\n")
        f.write(f"数据集: {dataset_path}\n")
        f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
        f.write(f"精确率 (Precision): {results.box.p:.4f}\n")
        f.write(f"召回率 (Recall): {results.box.r:.4f}\n")
        f.write(f"F1-Score: {2 * (results.box.p * results.box.r) / (results.box.p + results.box.r + 1e-16):.4f}\n")
    
    print(f"\n评估结果已保存到: {os.path.join(result_path, 'metrics.txt')}")
    print("评估完成!")

if __name__ == "__main__":
    main() 