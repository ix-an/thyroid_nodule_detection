import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score


def calculate_metrics(labels, preds):
    """计算各种评估指标

    Args:
        labels: 真实标签
        preds: 预测标签

    Returns:
        包含各种评估指标的字典
    """
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # 计算各项指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # 计算AUC（需要预测概率，这里假设preds是概率）
    # 如果preds是类别标签，需要获取模型的预测概率才能计算AUC
    # auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        # "auc": auc
    }