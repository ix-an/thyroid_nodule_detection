import torch                          # PyTorch库
import torch.utils.data as data       # PyTorch数据加载工具
from torchvision import transforms    # PyTorch视觉库中的图像转换工具
from PIL import Image                 # Python Imaging Library，用于处理图像
import numpy as np
import pandas as pd
import os

class ThyroidDataset(data.Dataset):
    """
    甲状腺结节图像数据集加载器
    用于加载甲状腺超声图像并关联对应的分类标签（良性/恶性）
    """
    def __init__(self, csv_path, image_dir, transform=None):
        """
        初始化甲状腺结节数据集
        Args:
            csv_path (str): 包含图像路径和标签的CSV文件路径
            image_dir (str): 包含图像的目录路径
            transform (callable, optional): 应用于图像的可选转换
        """
        # 读取CSV文件
        try:
            # 指定列名，且忽略第一行
            self.data_frame = pd.read_csv(csv_path,
                                          header=None,
                                          names=['image_id', 'label'],
                                          skiprows=1)
            print(f"✅ 成功加载 {len(self.data_frame)} 条数据")
        except FileNotFoundError as e:
            print(f"❌ 加载CSV文件时出错：{e}")
            raise

        self.image_dir = image_dir
        self.transform = transform
        # 标签映射：良性(benign)为0，恶性(malignant)为1
        self.label_map = {0: 0, 1: 1}



    def  __len__(self):
        """
        返回数据集的长度（样本数量）
        """
        return len(self.data_frame)


    def  __getitem__(self, idx):
        """
        根据索引返回一个数据样本
        Args:
            idx (int): 样本的索引
        Returns:
            tuple (image, label): 包含图像和标签的元组
        """
        # 获取图像文件名和标签
        image_id = self.data_frame.iloc[idx]['image_id']
        label_text = self.data_frame.iloc[idx]['label']

        # 将文本标签转换为数值标签
        label = self.label_map[label_text]

        # 构建完整的图像路径并加载图像
        image_path = os.path.join(self.image_dir, image_id)

        # 打开图像并转换为RGB模式
        # 即使图像本身是灰度图或二值图，转换为RGB模式可以确保数据一致性
        # 大多数预训练模型（如ResNet、VGG等）期望输入是RGB三通道图像
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"❌ 图像文件未找到：{image_path}")
            # 返回空白图像和无效标签
            return torch.zeros(3, 224, 224), -1

        # 应用图像转换（如调整大小、归一化、数据增强等）
        if self.transform:
            image = self.transform(image)

        return image, label

