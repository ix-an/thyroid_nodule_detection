"""数据集与数据增强模块"""

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os


class ThyroidDataset(data.Dataset):
    """甲状腺结节图像数据集加载器，包含增强的数据增强功能"""

    def __init__(self, csv_path, image_dir, transform=None):
        """
        初始化甲状腺结节数据集
        Args:
            csv_path (str): 包含图像路径和标签的CSV文件路径
            image_dir (str): 包含图像的目录路径
            transform (callable, optional): 应用于图像的可选转换
        """
        try:
            # 读取CSV文件，设置表头并跳过第一行
            self.data_frame = pd.read_csv(csv_path, header=None, names=['image_id', 'label'], skiprows=1)
            print(f"✅ 成功加载 {len(self.data_frame)} 条数据")
        except FileNotFoundError as e:
            print(f"❌ 加载CSV文件时出错：{e}")
            raise

        self.image_dir = image_dir       # 图像存储路径
        self.transform = transform       # 数据转换函数
        self.label_map = {0: 0, 1: 1}    # 标签映射：良性为0，恶性为1

        # 统计类别分布
        self.class_distribution = self.data_frame['label'].value_counts().to_dict()
        print(f"类别分布: 良性({self.class_distribution.get(0, 0)}) 恶性({self.class_distribution.get(1, 0)})")


    def __len__(self):
        """返回数据集的总样本数，用于DataLoader批量加载"""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        根据索引获取单个数据样本
        Args：
            idx (int): 样本索引
        Returns:
            tuple: (图像张量, 标签)，若图像加载失败则返回全0张量和-1标签
        """
        image_id = self.data_frame.iloc[idx]['image_id']    # 获取图像ID
        label_text = self.data_frame.iloc[idx]['label']     # 获取原始标签
        label = self.label_map[label_text]                  # 映射为模型标签

        # 拼接图像路径
        image_path = os.path.join(self.image_dir, image_id)
        try:
            # 打开图像并转换为RGB格式（兼容灰度图）
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"❌ 图像文件未找到：{image_path}")
            # 返回默认张量（错误处理）
            return torch.zeros(3, 224, 224), -1

        # 应用数据转换
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义数据转换函数：训练时使用增强，验证时使用基础转换
def get_data_transforms(image_size=224, is_train=True):
    """
    获取数据预处理与增强管道
    Args:
        image_size (int): 输出图像尺寸
        is_train (bool): 是否为训练模式（训练时启用增强，验证时仅标准化）
    Returns:
        transforms.Compose: 组合的数据转换操作
    """
    if is_train:
        """训练集增强策略：多种变换扩充数据，防止过拟合"""
        return transforms.Compose([
            # 先放大再随机裁剪，增加尺度不变性
            transforms.Resize((image_size + 30, image_size + 30)),
            transforms.RandomCrop((image_size, image_size)),
            # 随机水平翻转（50%概率）和垂直翻转（30%概率）
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            # 随机旋转±15度，增强角度不变性
            transforms.RandomRotation(degrees=15),
            # 调整亮度、对比度、饱和度和色调，增强颜色鲁棒性
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # 转换为张量并标准化（使用ImageNet预训练均值和方差）
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        """验证集标准变换：仅需标准化，保持数据一致性"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])