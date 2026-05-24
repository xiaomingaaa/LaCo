import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, 
    roc_auc_score, cohen_kappa_score, average_precision_score, 
    precision_score, recall_score
)

def get_metrics(y_true, y_pred):
    # 对模型输出进行sigmoid处理，得到概率值
    y_pred_prob = torch.sigmoid(torch.tensor(y_pred))  # Sigmoid 获取概率
    y_pred_bin = (y_pred_prob > 0.5).float()  # 将概率转化为二分类标签 (0 或 1)

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred_bin)
    
    # 计算 F1 分数
    f1 = f1_score(y_true, y_pred_bin)
    
    # 计算 MCC (Matthews 相关系数)
    mcc = matthews_corrcoef(y_true, y_pred_bin)
    
    # 计算 ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_prob)  # 使用概率值计算 ROC AUC
    
    # 计算 Kappa 系数
    kappa = cohen_kappa_score(y_true, y_pred_bin)
    
    # 计算平均精度 (AP)
    ap = average_precision_score(y_true, y_pred_prob)  # 使用概率值计算 AP
    
    # 计算精确率和召回率
    precision = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)

    return accuracy, f1, mcc, roc_auc, kappa, ap, precision, recall
