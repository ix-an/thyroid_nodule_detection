import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=2, model_name="resnet18", pretrained=True):
    """
    创建甲状腺结节分类模型（使用预训练模型进行迁移学习）

    Args:
        num_classes: 分类类别数（良性/恶性为2类）
        model_name: 预训练模型名称，支持'resnet18', 'vgg16'
        pretrained: 是否使用预训练权重
    """
    if model_name == "resnet18":
        # 加载ResNet18预训练模型
        model = models.resnet18(pretrained=pretrained)
        # 修改最后一层全连接层以适应2分类任务
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        # 加载VGG16预训练模型
        model = models.vgg16(pretrained=pretrained)
        # 修改最后一层全连接层
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model


# 测试模型创建
if __name__ == "__main__":
    # 创建2分类模型（良性/恶性）
    model = create_model(num_classes=2, model_name="resnet18")
    print(model)  # 打印模型结构，确认最后一层已修改

    # 测试其他模型
    model_vgg = create_model(num_classes=2, model_name="vgg16")
    print("\nVGG16模型:")
    print(model_vgg)