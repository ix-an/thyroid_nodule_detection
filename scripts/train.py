"""训练模块"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from models.model import create_model
from utils.data_utils import create_data_loaders
import numpy as np


def train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=15,
        learning_rate=0.0001,
        weight_decay=1e-5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path="../results/models",
        model_name="resnet50"
):
    """
    训练甲状腺结节分类模型，包含增强的训练策略和早停机制
    """
    os.makedirs(save_path, exist_ok=True)

    # 处理类别不平衡（计算类别权重）
    if hasattr(train_loader.dataset, 'class_distribution'):
        class_counts = np.array([train_loader.dataset.class_distribution.get(0, 1),
                                 train_loader.dataset.class_distribution.get(1, 1)])
        class_weights = 1.0 / class_counts
        class_weights = class_weights / np.sum(class_weights) * 2  # 归一化
        weight_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"使用类别权重: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()

    # 优化器与学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2,
        verbose=True, min_lr=1e-6
    )

    model = model.to(device)
    print(f"使用设备: {device}, 模型: {model_name}")

    best_val_acc = 0.0
    best_model_path = os.path.join(save_path, f"best_{model_name}.pth")
    early_stopping_counter = 0
    early_stopping_limit = 10  # 早停阈值

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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

        with torch.no_grad(), tqdm(val_loader, desc="验证中") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                pbar.set_postfix(loss=loss.item(), acc=100.0 * val_correct / val_total)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * val_correct / val_total
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        print(f"验证损失: {epoch_val_loss:.4f}, 准确率: {epoch_val_acc:.2f}%")

        # 更新学习率
        scheduler.step(epoch_val_acc)

        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "accuracy": best_val_acc,
                "history": history,
                "model_name": model_name
            }, best_model_path)
            print(f"✅ 保存最佳模型，验证准确率: {best_val_acc:.2f}%")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"🔄 未改进，早停计数器: {early_stopping_counter}/{early_stopping_limit}")

        # 早停机制
        if early_stopping_counter >= early_stopping_limit:
            print(f"⚠️ 达到早停条件，提前终止训练")
            break

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    return model, history


if __name__ == "__main__":
    # 数据路径（根据实际情况修改）
    train_csv = "../data/Thyroid_nodule_Dataset/label4train.csv"
    val_csv = "../data/Thyroid_nodule_Dataset/label4test.csv"
    train_dir = "../data/Thyroid_nodule_Dataset/train-image"
    val_dir = "../data/Thyroid_nodule_Dataset/test-image"

    # 创建数据加载器（增加batch_size和num_workers）
    train_loader, val_loader = create_data_loaders(
        train_csv=train_csv,
        val_csv=val_csv,
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=32,
        image_size=256,  # 增大图像尺寸
        num_workers=4  # 根据CPU核心数调整
    )

    # 创建模型（使用更深的ResNet50）
    model = create_model(num_classes=2, model_name="resnet50")
    print(model)

    # 训练模型（增加训练轮数，降低初始学习率）
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=0.0001
    )