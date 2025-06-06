import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from models.model import create_model
from utils.data_utils import create_data_loaders
from utils.metrics import calculate_metrics
import numpy as np


def train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=30,
        learning_rate=0.0001,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path="../results/models"
):
    os.makedirs(save_path, exist_ok=True)

    # 计算类别权重
    train_dataset = train_loader.dataset
    label_counts = train_dataset.data_frame['label'].value_counts()
    class_weights = torch.tensor(
        [label_counts[1] / label_counts[0], 1.0] if label_counts[0] > label_counts[1] else
        [1.0, label_counts[0] / label_counts[1]],
        dtype=torch.float, device=device
    )
    print(f"类别权重: {class_weights.tolist()}")

    # 使用焦点损失
    from utils.losses import FocalLoss
    criterion = FocalLoss(alpha=class_weights, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model = model.to(device)
    print(f"使用设备: {device}")

    best_val_acc = 0.0
    best_model_path = os.path.join(save_path, "best_model.pth")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "sensitivity": [],
        "specificity": []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        with tqdm(train_loader, desc="训练中") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (preds == labels).sum().item()

                pbar.set_postfix(loss=loss.item(), acc=100.0 * train_correct / train_total)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * train_correct / train_total
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        print(f"训练损失: {epoch_train_loss:.4f}, 准确率: {epoch_train_acc:.2f}%")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad(), tqdm(val_loader, desc="验证中") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

                pbar.set_postfix(loss=loss.item(), acc=100.0 * val_correct / val_total)

        val_metrics = calculate_metrics(np.array(all_val_labels), np.array(all_val_preds))
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * val_correct / val_total
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["sensitivity"].append(val_metrics["sensitivity"])
        history["specificity"].append(val_metrics["specificity"])

        print(f"验证损失: {epoch_val_loss:.4f}, 准确率: {epoch_val_acc:.2f}%")
        print(f"敏感性: {val_metrics['sensitivity']:.4f}, 特异性: {val_metrics['specificity']:.4f}")

        scheduler.step()

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "accuracy": best_val_acc,
                "history": history
            }, best_model_path)
            print(f"✅ 保存最佳模型，验证准确率: {best_val_acc:.2f}%")

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    return model, history


if __name__ == "__main__":
    # 数据路径
    train_csv = "../data/Thyroid_nodule_Dataset/label4train.csv"
    val_csv = "../data/Thyroid_nodule_Dataset/label4test.csv"
    train_dir = "../data/Thyroid_nodule_Dataset/train-image"
    val_dir = "../data/Thyroid_nodule_Dataset/test-image"

    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        train_csv=train_csv,
        val_csv=val_csv,
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=8,  # 降低批量大小适应CPU或小显存
        image_size=224
    )

    # 创建模型（使用新的weights参数）
    model = create_model(num_classes=2, model_name="resnet50")

    # 仅解冻最后两个残差块
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 打印可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

    # 训练模型
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,
        learning_rate=0.0001
    )