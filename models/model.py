"""模型定义模块"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=2, model_name="resnet50", pretrained=True, use_pretrained=True):
    """
    创建甲状腺结节分类模型，支持多种预训练模型和集成选项
    Args:
        num_classes: 分类类别数（良性/恶性为2类）
        model_name: 预训练模型名称，支持'resnet18', 'resnet50', 'vgg16', 'densenet121'
        pretrained: 是否使用预训练权重
        use_pretrained: 是否冻结底层参数，仅训练上层
    """
    # 根据不同的模型名称选择对应的预训练模型
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        # 修改最后一层全连接层，使其输出维度等于分类类别数
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        # 修改VGG16的最后一层全连接层
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}，支持的模型: resnet18, resnet50, vgg16, densenet121")

    # 冻结部分底层参数，只训练上层
    # 预训练模型的底层参数通常学习到了一些通用的特征，如边缘、纹理等。
    # 冻结这些参数可以加快训练速度，同时避免过拟合。只训练上层参数可以让模型更快地适应新的数据集。
    if use_pretrained:
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后几层进行训练
        if "resnet" in model_name:
            # 解冻ResNet的最后一个残差块和全连接层
            for param in model.layer4.parameters():
                param.requires_grad = True
            model.fc.requires_grad = True
        elif model_name == "vgg16":
            # 解冻VGG16的最后两层全连接层
            for param in model.classifier[4:].parameters():
                param.requires_grad = True
        elif model_name == "densenet121":
            # 解冻DenseNet121的分类器和部分特征层
            for param in model.classifier.parameters():
                param.requires_grad = True
                for param in model.features.named_parameters():
                    if 'transition' in param[0] or 'norm5' in param[0] or 'relu5' in param[0] or 'conv5' in param[0]:
                        param[1].requires_grad = True

    return model


# 模型集成类（可选，用于提升性能）
class ModelEnsemble(nn.Module):
    def __init__(self, models):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        # 简单平均集成
        return sum(outputs) / len(outputs)