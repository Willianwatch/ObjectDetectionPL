import torch.nn as nn
import torch


class FPN(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(FPN, self).__init__()
        # ! here is no bathcnorm?? relu only p7
        # upsample C5 to get P5 from the FPN paper
        # C5 to P5做了1*1卷积
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # 最近邻上采样
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # 不改变shape
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        # if p6 use the P5 ,just change the argu C5_size to 256 !!!
        # self.P6 = nn.Conv2d(256, feature_size, kernel_size=3, stride=2, padding=1)
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)



        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        # P5 here is just use c5 -> 1 * 1 ->3*3(no change the shape)
        # just change the channel to 256
        P5_x = self.P5_1(C5)
        # p5 upsampled_x is for the P4 use
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        # 是针对C5进行的，这里也可以改成正对P5进行的
        # P6_x = self.P6(P5_x)
        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
    
if __name__ == '__main__':
    C3 = torch.randn([2,16,200,200])
    C4 = torch.randn([2,32,100,100])
    C5 = torch.randn([2,64,50,50])

    model = FPN(16,32,64)

    out = model([C3,C4,C5])
    for i in range(len(out)):
        print(out[i].shape)
    # torch.Size([2, 256, 200, 200])
    # torch.Size([2, 256, 100, 100])
    # torch.Size([2, 256, 50, 50])
    # torch.Size([2, 256, 25, 25])
    # torch.Size([2, 256, 13, 13])