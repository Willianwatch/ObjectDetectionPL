import torch.nn as nn
import torch
class RegressionModel(nn.Module):
    # 最后要得出 4 * num_anchors，也就是每一个anchor的4个值
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        # 不变shape,还是没有batchnorm，添加也许有点用。4个一样的shape，最后一个channel改变
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        # 改变顺序b w h c
        out = out.permute(0, 2, 3, 1)
        # 通过contiguout().view变成 b , w*h, 4
        return out.contiguous().view(out.shape[0], -1, 4)
    

if __name__ == '__main__':
    C = torch.randn([2,256,512,512])
    model = RegressionModel(256)
    out = model(C)
    print(out.shape)
    for i in range(len(out)):
        print(out[i].shape)
    # 说明用len可以得出第一维
    # torch.Size([2, 2359296, 4])
    # torch.Size([2359296, 4])
    # torch.Size([2359296, 4])