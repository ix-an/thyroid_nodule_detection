import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from utils.datasets import ThyroidDataset    # 导入自定义数据集类
from torch.utils.data import DataLoader
import pandas as pd

def test_thyroid_dataset():
    # 定义测试用的数据转换（仅调整尺寸和转张量，不增强）
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 指定数据集路径
    csv_path = '../data/Thyroid_nodule_Dataset/label4train.csv'
    image_dir = '../data/Thyroid_nodule_Dataset/train-image'

    # 计算预期数据集长度（CSV文件行数）
    expected_length = len(pd.read_csv(csv_path))

    # 实例化数据集对象
    dataset = ThyroidDataset(csv_path, image_dir, transform=test_transform)


    # 测试 1: 检查数据集长度是否与CSV文件行数一致
    print("数据集长度测试:", len(dataset) == expected_length)

    # 测试 2: 获取第一个样本，检查标签和图像形状
    image, label = dataset[0]
    print("第一个样本标签测试:", label in [0, 1])
    print("第一个样本图像形状测试:", image.shape == torch.Size([3, 224, 224]))

    # 测试 3: 获取第二个样本，重复验证（确保非偶然正确）
    image2, label2 = dataset[1]
    print("第二个样本标签测试:", label2 in [0, 1])
    print("第二个样本图像形状测试:", image2.shape == torch.Size([3, 224, 224]))

    # 测试 4: 验证批量加载功能（batch_size=4）
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for images, labels in dataloader:
        print("批量标签长度测试:", len(labels) == 4)
        print("批量图像形状测试:", images.shape == torch.Size([4, 3, 224, 224]))
        break  # 仅检查第一个批量，避免重复输出

if __name__ == "__main__":
    test_thyroid_dataset()