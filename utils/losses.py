import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """焦点损失函数，用于处理类别不平衡问题

    论文: https://arxiv.org/abs/1708.02002

    Args:
        alpha: 类别权重张量
        gamma: 聚焦参数，调整难易样本的权重
        reduction: 损失计算方式，可选'mean', 'sum', 'none'
    """

    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # 计算预测概率
        pt = torch.exp(-ce_loss)

        # 计算焦点损失
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 应用规约方式
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss