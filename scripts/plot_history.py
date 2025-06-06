import torch
import matplotlib.pyplot as plt
import os


def plot_training_history(history):
    """
    绘制训练损失和准确率曲线
    """
    plt.figure(figsize=(12, 5))

    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="训练损失", color="blue")
    plt.plot(history["val_loss"], label="验证损失", color="red")
    plt.title("损失曲线")
    plt.xlabel("轮次(Epoch)")
    plt.ylabel("损失值(Loss)")
    plt.legend()
    plt.grid(True)

    # 绘制训练准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="训练准确率", color="blue")
    plt.plot(history["val_acc"], label="验证准确率", color="red")
    plt.title("准确率曲线")
    plt.xlabel("轮次(Epoch)")
    plt.ylabel("准确率(Accuracy)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = "../results/history/resnet18_history.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    print(f"训练历史图已保存到: {save_path}")


if __name__ == "__main__":
    # 加载模型检查点
    model_path = "../results/models/best_model.pth"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    if "history" in checkpoint:
        history = checkpoint["history"]
        plot_training_history(history)
    else:
        print("警告: 检查点中未包含训练历史，无法绘制曲线")