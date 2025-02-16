import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
import numpy as np
import utils.datasets
from torch.utils.data import DataLoader

def genertate_region_mask(masks ,batch_shape):

    # print(f"batch_shape的维度是{batch_shape.shape}")

    """generate B 1 H W meaningful-region-mask for a batch of masks

    Args:
        batch_shape (_type_): _description_
    """
    meaningful_mask = torch.zeros_like(masks)
    for idx, shape in enumerate(batch_shape):
        # print(shape.shape)
        # print(shape[0])
        # print(shape[1])
        meaningful_mask[idx, :, :shape[0], :shape[1]] = 1
    return meaningful_mask

# def genertate_region_mask(masks ,batch_shape):
#     """generate B 1 H W meaningful-region-mask for a batch of masks

#     Args:
#         batch_shape (_type_): _description_
#     """
#     meaningful_mask = torch.zeros_like(masks)
#     # print(f"meaningful_mask shape: {meaningful_mask.shape}")  # 检查初始化后的形状   meaningful_mask shape: torch.Size([1, 1, 1024, 1024])
#     for idx, shape in enumerate(batch_shape):
#         # print(f"idx: {idx}, shape: {shape}")  # 打印当前索引和 shape 值   idx: 0, shape: tensor([[ 0.0224, -0.0047, -0.0147,  ...,  0.0084, -0.0176, -0.0027]])
#         # print(f"Length of batch shape: {len(batch_shape)}")    #Length of batch shape: 1
#         if shape.dim() == 2:  # 假设 shape 是 2D 张量
#             height = shape.size(0)  # 提取高度
#             width = shape.size(1)  # 提取宽度
#         else:
#             height, width = shape[0], shape[1]  # 处理正常的 [H, W] 形式

#         meaningful_mask[idx, :, :height, :width] = 1

#         # meaningful_mask[idx, :, :shape[0], :shape[1]] = 1
#         print(f"Final meaningful_mask: {meaningful_mask}")
#     return meaningful_mask

def cal_confusion_matrix(predict, target, region_mask, threshold=0.5):
    """compute local confusion matrix for a batch of predict and target masks
    Args:
        predict (_type_): _description_
        target (_type_): _description_
        region (_type_): _description_
        
    Returns:
        TP, TN, FP, FN
    """
    predict = (predict > threshold).float()
    TP = torch.sum(predict * target * region_mask, dim=(1, 2, 3))
    TN = torch.sum((1-predict) * (1-target) * region_mask, dim=(1, 2, 3))
    FP = torch.sum(predict * (1-target) * region_mask, dim=(1, 2, 3))
    FN = torch.sum((1-predict) * target * region_mask, dim=(1, 2, 3))
    return TP, TN, FP, FN

def cal_F1(TP, TN, FP, FN):
    """compute F1 score for a batch of TP, TN, FP, FN
    Args:
        TP (_type_): _description_
        TN (_type_): _description_
        FP (_type_): _description_
        FN (_type_): _description_
    """
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    # F1 = torch.mean(F1) # fuse the Batch dimension
    return F1

###################################################################################
def cal_precise_AUC_with_shape(predict, target, shape):
    predict2 = predict[0][0][:shape[0][0], :shape[0][1]]
    target2 = target[0][0][:shape[0][0], :shape[0][1]]
    # flat to single dimension fit the requirements of the sklearn 
    predict3 = predict2.reshape(-1).cpu()
    target3 = target2.reshape(-1).cpu()
    # -----visualize roc curve-----
    fpr, tpr, thresholds = roc_curve(target3, predict3, pos_label=1)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig("./appro2.png")
    # ------------------------------
    AUC = roc_auc_score(target3, predict3)
    return AUC
##################################################################################
    