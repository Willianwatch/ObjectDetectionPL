import torch.nn as nn
import torch

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # 一样的3个同样的不改变shape的卷积，无bn
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        # 这里和regression区别在于*num_classes
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        # 还要进行一次sigmoid，为了不出现负数，将其映射到0-1之间
        self.output_act = nn.Sigmoid()

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
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
    

if __name__ == '__main__':
    C1 = torch.randn([2, 256, 64, 64])
    C2 = torch.randn([2, 256, 32, 32])
    C3 = torch.randn([2, 256, 16, 16])
    C4 = torch.randn([2, 256, 8, 8])
    C5 = torch.randn([2, 256, 4, 4])
    model = ClassificationModel(256)
    print(model(C1).shape)
    print(model(C2).shape)
    print(model(C3).shape)
    print(model(C4).shape)
    print(model(C5).shape)

    # torch.Size([2, 36864, 80])
    # torch.Size([2, 9216, 80])
    # torch.Size([2, 2304, 80])
    # torch.Size([2, 576, 80])
    # torch.Size([2, 144, 80])