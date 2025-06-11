"""训练历史可视化模块"""

import torch
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_training_history(history, model_name="resnet50"):
    """绘制训练损失和准确率曲线，增加更多评估指标可视化"""
    plt.figure(figsize=(16, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="训练损失", color="blue", linewidth=2)
    plt.plot(history["val_loss"], label="验证损失", color="red", linewidth=2)
    plt.title("损失曲线", fontsize=14)
    plt.xlabel("轮次(Epoch)", fontsize=12)
    plt.ylabel("损失值(Loss)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="训练准确率", color="blue", linewidth=2)
    plt.plot(history["val_acc"], label="验证准确率", color="red", linewidth=2)
    plt.title("准确率曲线", fontsize=14)
    plt.xlabel("轮次(Epoch)", fontsize=12)
    plt.ylabel("准确率(%)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    save_path = f"../results/history/{model_name}_history.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"训练历史图已保存到: {save_path}")

    # 额外输出最佳验证准确率
    best_epoch = np.argmax(history["val_acc"])
    print(f"最佳验证准确率出现在第 {best_epoch + 1} 轮: {history['val_acc'][best_epoch]:.2f}%")


if __name__ == "__main__":
    # 模型路径
    model_path = "../results/models/best_resnet50.pth"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    if "history" in checkpoint:
        history = checkpoint["history"]
        model_name = checkpoint.get("model_name", "resnet50")
        plot_training_history(history, model_name)
    else:
        print("警告: 检查点中未包含训练历史，无法绘制曲线")