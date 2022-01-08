import torch
import torch.nn as nn


# 裁切我们的bbox
class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape
        # 下面就是我找的torch官网的关于clamp函数的介绍！
        # 其实就是小于0的x1,y1变为0，大于w,h的x2,y2变为w,h
        # Clamps all elements in input into the range[min, max].Letting min_value and max_value be min and max
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes