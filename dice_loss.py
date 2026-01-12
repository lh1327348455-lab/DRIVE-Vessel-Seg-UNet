import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 1. 经过 Sigmoid 激活，因为模型输出是 Logits
        inputs = F.sigmoid(inputs)       
        
        # 2. 展平 (Flatten)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 3. 计算 Dice 系数
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        # 4. 返回 Loss (1 - Dice)
        return 1 - dice