import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from models.model import create_model
from utils.data_utils import create_data_loaders


def train_model(
        model,                                                    # 要训练的模型
        train_loader,                                             # 训练数据加载器
        val_loader,                                               # 验证数据加载器
        num_epochs=10,                                            # 训练轮数
        learning_rate=0.001,                                      # 学习率
        device="cuda" if torch.cuda.is_available() else "cpu",    # 训练设备（'cuda'或'cpu'）
        save_path="../results/models"                             # 模型保存路径
):
    """
    训练甲状腺结节分类模型，并保存训练过程中的损失和准确率
    """
    # 确认目录存在
    os.makedirs(save_path, exist_ok=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()    # 多分类交叉熵损失，适用于2分类
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # 模型移至指定设备
    model = model.to(device)
    print(f"使用设备: {device}")

    # 记录最佳模型
    best_val_acc = 0.0
    best_model_path = os.path.join(save_path, "best_model.pth")

    # 训练历史
    history =  {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 开始训练
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

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 记录损失和准确率
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (preds == labels).sum().item()

                # 更新进度条
                pbar.set_postfix(loss=loss.item(), acc=100.0 * train_correct / train_total)

        # 计算平均损失和准确率
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
        scheduler.step(epoch_val_loss)

        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "accuracy": best_val_acc
            }, best_model_path)
            print(f"✅ 保存最佳模型，验证准确率: {best_val_acc:.2f}%")

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    return model, history


if __name__ == "__main__":
    # 数据路径（根据实际情况修改）
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
        batch_size=16,
        image_size=224
    )

    # 创建模型
    model = create_model(num_classes=2, model_name="resnet18")
    print(model)

    # 训练模型（初始训练轮数设为5-10轮，避免过拟合）
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=0.001
    )