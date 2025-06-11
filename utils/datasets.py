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
            self.data_frame = pd.read_csv(csv_path, header=None, names=['image_id', 'label'], skiprows=1)
            print(f"✅ 成功加载 {len(self.data_frame)} 条数据")
        except FileNotFoundError as e:
            print(f"❌ 加载CSV文件时出错：{e}")
            raise
        self.image_dir = image_dir
        self.transform = transform
        # 标签映射：良性为0，恶性为1
        self.label_map = {0: 0, 1: 1}

        # 统计类别分布
        self.class_distribution = self.data_frame['label'].value_counts().to_dict()
        print(f"类别分布: 良性({self.class_distribution.get(0, 0)}) 恶性({self.class_distribution.get(1, 0)})")

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """根据索引返回一个数据样本"""
        image_id = self.data_frame.iloc[idx]['image_id']
        label_text = self.data_frame.iloc[idx]['label']
        label = self.label_map[label_text]

        image_path = os.path.join(self.image_dir, image_id)
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"❌ 图像文件未找到：{image_path}")
            return torch.zeros(3, 224, 224), -1

        if self.transform:
            image = self.transform(image)
        return image, label


# 定义增强的数据转换函数
def get_data_transforms(image_size=224, is_train=True):
    """获取数据转换，训练时使用增强，验证时使用基础转换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 30, image_size + 30)),  # 先放大再裁剪，增加随机性
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])