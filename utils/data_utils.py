"""数据加载工具模块"""

import torch
from torch.utils.data import DataLoader
from utils.datasets import ThyroidDataset, get_data_transforms    # 导入自定义数据集和数据增强模块


def create_data_loaders(train_csv, val_csv, train_dir, val_dir,
                        batch_size=32, image_size=224, num_workers=4):
    """
    创建训练集和验证集的数据加载器
    Args:
        train_csv (str): 训练集CSV文件路径
        val_csv (str): 验证集CSV文件路径
        train_dir (str): 训练集图像目录
        val_dir (str): 验证集图像目录
        batch_size (int): 批量大小
        image_size (int): 图像尺寸
        num_workers (int): 数据加载的工作进程数
    Returns:
        tuple: (训练数据加载器, 验证数据加载器)
    """
    # 获取训练集和验证集的数据转换函数
    # 训练集使用数据增强，验证集仅做标准化
    train_transform = get_data_transforms(image_size, is_train=True)
    val_transform = get_data_transforms(image_size, is_train=False)

    # 实例化数据集对象
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

    # 创建数据加载器 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,               # 训练集洗牌
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,              # 验证集不洗牌
        num_workers=num_workers,
        pin_memory=True
    )

    # 打印数据集统计信息（调试用）
    print(f"训练数据加载器: {len(train_loader)} 个批次，共 {len(train_dataset)} 样本")
    print(f"验证数据加载器: {len(val_loader)} 个批次，共 {len(val_dataset)} 样本")
    return train_loader, val_loader    # 返回两个数据加载器


def create_validation_loader(val_csv, val_dir, batch_size=32, image_size=224, num_workers=4):
    """
    创建验证数据加载器,用于模型评估
    逻辑与create_data_loaders中的验证集部分完全一致
    """
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