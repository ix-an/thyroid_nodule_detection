import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from utils.datasets import ThyroidDataset


# 修复后的高斯噪声变换（关键修改：返回PIL Image）
class GaussianNoise(object):
    def __init__(self, sigma=0.08, p=0.3):  # 降低噪声强度和应用概率
        """添加高斯噪声到图像

        Args:
            sigma: 噪声标准差，默认0.08
            p: 应用噪声的概率，默认0.3
        """
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        # 确保输入是Tensor
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)

        if np.random.rand() < self.p:
            # 添加高斯噪声
            noisy_img = img + torch.randn(img.size()) * self.sigma
            # 将Tensor转回PIL Image（关键修改）
            noisy_img = torch.clamp(noisy_img, 0, 1)  # 确保像素值在[0,1]之间
            return F.to_pil_image(noisy_img)
        else:
            # 如果不添加噪声，也转回PIL Image
            return F.to_pil_image(img)

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0}, p={1})'.format(self.sigma, self.p)


def get_data_transforms(image_size=224):
    """获取数据转换（按逻辑组拆分，提高可读性）"""
    # 基础变换（尺寸调整）
    base_transforms = [
        transforms.Resize((image_size + 40, image_size + 40)),
        transforms.RandomCrop(image_size),
    ]

    # 几何变换（位置、角度、缩放）
    geometric_transforms = [
        transforms.RandomHorizontalFlip(p=0.7),  # 水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 垂直翻转
        transforms.RandomRotation(degrees=20),  # 随机旋转
        transforms.RandomAffine(
            degrees=15,
            translate=None,
            scale=(0.9, 1.1),  # 缩放变换
            shear=10,  # 剪切变换
            fill=0  # 填充空白区域
        ),
        transforms.RandomPerspective(
            distortion_scale=0.1,
            p=0.5,
            fill=0  # 填充空白区域
        ),
    ]

    # 颜色变换（亮度、对比度等）
    color_transforms = [
        transforms.ColorJitter(
            brightness=0.3,  # 亮度变化范围
            contrast=0.3,  # 对比度变化范围
            saturation=0.2,  # 饱和度变化范围
            hue=0.1  # 色调变化范围
        ),
        transforms.GaussianBlur(
            kernel_size=3,  # 高斯核大小
            sigma=(0.1, 2.0)  # 高斯核标准差范围
        ),
    ]

    # 医学专用变换（模拟超声特性）
    medical_transforms = [
        GaussianNoise(sigma=0.08, p=0.3),  # 模拟超声斑点噪声（降低强度）
    ]

    # 最终处理（转换为Tensor并归一化）
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet数据集的均值
            std=[0.229, 0.224, 0.225]  # ImageNet数据集的标准差
        ),
    ]

    # 组合训练集的所有变换
    train_transform = transforms.Compose(
        base_transforms +
        geometric_transforms +
        color_transforms +
        medical_transforms +
        final_transforms
    )

    # 验证集只需基础尺寸调整和最终处理
    val_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size))] +
        final_transforms
    )

    return train_transform, val_transform


def create_data_loaders(train_csv, val_csv, train_dir, val_dir, batch_size=16, image_size=224):
    """创建数据加载器

    Args:
        train_csv: 训练集CSV文件路径
        val_csv: 验证集CSV文件路径
        train_dir: 训练图像文件夹路径
        val_dir: 验证图像文件夹路径
        batch_size: 批次大小
        image_size: 图像尺寸

    Returns:
        训练和验证数据加载器
    """
    train_transform, val_transform = get_data_transforms(image_size)

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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows设为0避免多进程错误
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"训练数据加载器: {len(train_loader)} 个批次")
    print(f"验证数据加载器: {len(val_loader)} 个批次")

    return train_loader, val_loader


def create_validation_loader(val_csv, val_dir, batch_size=16, image_size=224):
    """创建仅用于验证的数据加载器

    Args:
        val_csv: 验证集CSV文件路径
        val_dir: 验证图像文件夹路径
        batch_size: 批次大小
        image_size: 图像尺寸

    Returns:
        验证数据加载器
    """
    _, val_transform = get_data_transforms(image_size)

    val_dataset = ThyroidDataset(
        csv_path=val_csv,
        image_dir=val_dir,
        transform=val_transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"验证数据加载器: {len(val_loader)} 个批次")
    return val_loader