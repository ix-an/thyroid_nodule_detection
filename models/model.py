import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights  # 修复预训练模型加载方式


def create_model(num_classes=2, model_name="resnet50", pretrained=True):
    """
    创建用于甲状腺结节分类的模型

    Args:
        num_classes: 分类类别数，默认为2（良性/恶性）
        model_name: 模型架构名称，默认为resnet50
        pretrained: 是否使用预训练权重

    Returns:
        配置好的PyTorch模型
    """
    if model_name == "resnet50":
        # 新的预训练模型加载方式（修复警告）
        if pretrained:
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)

        # 修改全连接层以适应我们的分类任务
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "efficientnet_b0":
        # 同理修改其他模型
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model


def freeze_layers(model, freeze_ratio=0.7):
    """
    冻结模型的部分层以进行微调

    Args:
        model: 待冻结的模型
        freeze_ratio: 冻结比例，例如0.7表示冻结前70%的层

    Returns:
        冻结后的模型
    """
    total_layers = len(list(model.parameters()))
    layers_to_freeze = int(total_layers * freeze_ratio)

    print(f"冻结{layers_to_freeze}/{total_layers}层")

    # 冻结指定比例的层
    for i, param in enumerate(model.parameters()):
        if i < layers_to_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model