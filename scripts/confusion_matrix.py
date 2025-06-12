"""评估模块"""

import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils.data_utils import create_validation_loader
from models.model import create_model


def plot_confusion_matrix(model, val_loader, device, class_names, save_path=None):
    """绘制混淆矩阵并计算详细评估指标"""
    # 将模型设置为评估模式
    model.eval()
    all_labels = []
    all_preds = []

    # 禁用梯度计算
    with torch.no_grad():
        for inputs, labels in val_loader:
            # 将输入和标签移动到指定设备
            inputs, labels = inputs.to(device), labels.to(device)
            # 前向传播
            outputs = model(inputs)
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            # 收集真实标签
            all_labels.extend(labels.cpu().numpy())
            # 收集预测标签
            all_preds.extend(preds.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    # 使用 seaborn 绘制热力图，设置颜色映射为 Blues
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("甲状腺结节分类混淆矩阵")
    plt.xticks(ticks=[0, 1], labels=class_names)
    plt.yticks(ticks=[0, 1], labels=class_names)
    plt.tight_layout()

    if save_path is None:
        # 默认模型名称
        model_name = "resnet50"
        save_path = f"../results/history/{model_name}_CM.png"
    # 创建保存路径的目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存混淆矩阵
    plt.savefig(save_path)
    plt.show()
    print(f"混淆矩阵已保存至: {save_path}")

    # 计算详细评估指标
    # 解包混淆矩阵元素
    tp, fp, fn, tn = cm.ravel()
    # 计算准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    # 计算敏感性（真阳性率）
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    # 计算特异性（真阴性率）
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # 计算精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # 计算 F1 分数
    f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0  # F1分数

    print(f"准确率: {accuracy:.4f}")
    print(f"敏感性(真阳性率): {sensitivity:.4f}")
    print(f"特异性(真阴性率): {specificity:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    print(f"真阳性(tp): {tp}, 真阴性(tn): {tn}")
    print(f"假阳性(fp): {fp}, 假阴性(fn): {fn}")

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def export_predictions(model, data_loader, device, output_path):
    """导出预测结果到CSV文件"""
    model.eval()
    results = []
    # 禁用梯度计算
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            results.extend(zip(labels.cpu().numpy(), preds.cpu().numpy()))

    df = pd.DataFrame(results, columns=['true_label', 'predicted_label'])
    df.to_csv(output_path, index=False)
    print(f"预测结果已导出至: {output_path}")


if __name__ == "__main__":
    # 数据路径
    val_csv = "../data/Thyroid_nodule_Dataset/label4test.csv"
    val_dir = "../data/Thyroid_nodule_Dataset/test-image"
    # 模型路径
    model_path = "../results/models/best_resnet50.pth"

    # 创建验证数据加载器（增大图像尺寸与batch_size一致）
    val_loader = create_validation_loader(
        val_csv=val_csv,
        val_dir=val_dir,
        batch_size=32,
        image_size=256
    )

    # 加载模型
    model = create_model(num_classes=2, model_name="resnet50")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 定义类别名称
    class_names = ["良性", "恶性"]

    # 绘制混淆矩阵并计算指标
    metrics = plot_confusion_matrix(model, val_loader, device, class_names=class_names)
    # 导出预测结果
    export_predictions(model, val_loader, device, "../results/history/resnet50_predictions.csv")