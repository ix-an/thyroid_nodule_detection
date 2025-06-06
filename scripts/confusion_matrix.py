import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils.data_utils import create_data_loaders, create_validation_loader
from models.model import create_model


def plot_confusion_matrix(model, val_loader, device, class_names, save_path=None):
    """
    绘制混淆矩阵
    """
    model.eval()  # 设置模型为评估模式
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())


    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("甲状腺结节分类混淆矩阵")
    plt.xticks(ticks=[0, 1], labels=class_names)
    plt.yticks(ticks=[0, 1], labels=class_names)
    plt.tight_layout()

    if save_path is None:
        save_path = "../results/history/resnet18_CM.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    print(f"混淆矩阵已保存至: {save_path}")

    # 计算评估指标
    tp, fp, fn, tn = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 真阳性率
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 真阴性率

    print(f"准确率: {accuracy:.4f}")
    print(f"敏感性(真阳性率): {sensitivity:.4f}")
    print(f"特异性(真阴性率): {specificity:.4f}")
    print(f"真阳性(tp): {tp}, 真阴性(tn): {tn}")
    print(f"假阳性(fp): {fp}, 假阴性(fn): {fn}")

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


if __name__ == "__main__":
    # 数据路径
    val_csv = "../data/Thyroid_nodule_Dataset/label4test.csv"
    val_dir = "../data/Thyroid_nodule_Dataset/test-image"

    # 模型路径
    model_path = "../results/models/best_model.pth"

    # 创建验证数据加载器
    from utils.data_utils import create_validation_loader

    val_loader = create_validation_loader(
        val_csv=val_csv,
        val_dir=val_dir,
        batch_size=16,
        image_size=224
    )

    # 加载模型
    model = create_model(num_classes=2, model_name="resnet18")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 定义类别名称
    class_names = ["良性", "恶性"]

    # 绘制混淆矩阵并计算指标
    metrics = plot_confusion_matrix(model, val_loader, device, class_names=class_names)