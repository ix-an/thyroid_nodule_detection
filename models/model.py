"""模型定义模块"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=2, model_name="resnet50", pretrained=True, use_pretrained=True):
    """
    创建甲状腺结节分类模型，支持更多预训练模型和集成选项
    Args:
        num_classes: 分类类别数（良性/恶性为2类）
        model_name: 预训练模型名称，支持'resnet18', 'resnet50', 'vgg16', 'densenet121'
        pretrained: 是否使用预训练权重
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}，支持的模型: resnet18, resnet50, vgg16, densenet121")

    # 冻结部分底层参数，只训练上层
    if use_pretrained:
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后几层进行训练
        if "resnet" in model_name:
            for param in model.layer4.parameters():
                param.requires_grad = True
            model.fc.requires_grad = True
        elif model_name == "vgg16":
            for param in model.classifier[4:].parameters():
                param.requires_grad = True
        elif model_name == "densenet121":
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