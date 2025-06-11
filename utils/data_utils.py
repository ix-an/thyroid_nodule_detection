"""数据加载工具模块"""

import torch
from torch.utils.data import DataLoader
from utils.datasets import ThyroidDataset, get_data_transforms


def create_data_loaders(train_csv, val_csv, train_dir, val_dir,
                        batch_size=32, image_size=224, num_workers=4):
    """
    创建数据加载器，包含增强的训练数据转换和基础的验证数据转换
    """
    # 获取数据转换
    train_transform = get_data_transforms(image_size, is_train=True)
    val_transform = get_data_transforms(image_size, is_train=False)

    # 加载数据集
    train_dataset = ThyroidDataset(
        csv_path=train_csv,
        image_dir=train_dir,
        transform=train_transform
    )
    val_dataset = ThyroidDataset(
        csv_path=val_csv,
        image_dir=val_dir,
        transform=val_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"训练数据加载器: {len(train_loader)} 个批次，共 {len(train_dataset)} 样本")
    print(f"验证数据加载器: {len(val_loader)} 个批次，共 {len(val_dataset)} 样本")
    return train_loader, val_loader


def create_validation_loader(val_csv, val_dir, batch_size=32, image_size=224, num_workers=4):
    """创建验证数据加载器"""
    val_transform = get_data_transforms(image_size, is_train=False)
    val_dataset = ThyroidDataset(
        csv_path=val_csv,
        image_dir=val_dir,
        transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"验证数据加载器: {len(val_loader)} 个批次，共 {len(val_dataset)} 样本")
    return val_loader