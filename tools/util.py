import os
import numpy as np
import math
import torch

def missing_list():
    n = int(math.pow(2,4))
    list = [[],[],[],[]]
    for i in range(n):
        if i % 2 == 1:
            list[0].append(i)
        if (i % 4)//2 == 1:
            list[1].append(i)
        if (i % 8)//4 == 1:
            list[2].append(i)
        if i // 8 == 1:
            list[3].append(i)
    return list

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def pad_zero(s, length=3):
    s = str(s)
    assert len(s) < length + 1
    if len(s) < length:
        s = '0' * (length - len(s)) + s
    return s

def zscore(x):
    x = (x - x.mean()) / x.std()
    return x

def min_max(x):
    mi, ma = x.min(), x.max()
    x = (x - mi) / (ma - mi)
    return x

def percentile(x, prct):
    low, high = np.percentile(x, prct[0]), np.percentile(x, prct[1])
    x[x < low] = low
    x[x > high] = high
    return x

def parse_image_name(name):
    n = name.split('.')[0]
    mod, pid, index, pn = n.split('_')
    return mod, pid, index, pn, 'modality'+name[len(mod):]

def center_crop(img, size):
    h, w = img.shape
    x, y = (h - size) // 2, (w - size) // 2
    img_ = img[x: x+size, y: y+size]
    return img_

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 输入inputs和targets的形状为[N, C, H, W]，其中N是批大小，C是通道数，H和W是高度和宽度
        # 确保输入是浮点类型，因为整数类型不支持梯度回传
        inputs = inputs.float()
        targets = targets.float()

        # 将inputs通过sigmoid激活函数进行归一化，使其值位于[0, 1]之间
        # inputs = torch.sigmoid(inputs)

        # 展平inputs和targets，计算它们的交集和并集
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # 计算Dice损失
        dice_loss = 1 - dice

        # 返回损失的均值
        return dice_loss.mean()
