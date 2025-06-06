import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.datasets import ThyroidDataset


def get_data_transforms(image_size=224):
    """获取数据转换（简化版，暂不使用复杂增强）"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_data_loaders(train_csv, val_csv, train_dir, val_dir, batch_size=16, image_size=224):
    """创建数据加载器"""
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows设为0避免多进程错误
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"训练数据加载器: {len(train_loader)} 个批次")
    print(f"验证数据加载器: {len(val_loader)} 个批次")

    return train_loader, val_loader