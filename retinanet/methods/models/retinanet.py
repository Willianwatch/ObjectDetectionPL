import math
import time

import torch
import torch.nn as nn

from .backbones.resnet import resnet50
from .necks.fpn import FPN
from .heads.classification import ClassificationModel
from .heads.regression import RegressionModel
from .anchors import Anchors
from .utils.bboxtransform import BBoxTransform
from .utils.clipbox import ClipBoxes
from .utils.nms import nms


class RetinaNet(nn.Module):
    def __init__(self, num_classes):
        super(RetinaNet, self).__init__()
        self.resnet = resnet50(num_classes, pretrained=True)
        # 512，1024，2048是针对resnet50以上的channel数
        self.fpn = FPN(512,1024,2048)
        self.regression = RegressionModel(256)
        self.classification = ClassificationModel(256,num_classes=num_classes)
        # 初始化
        prior = 0.01
        self.classification.output.weight.data.fill_(0)
        self.classification.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regression.output.weight.data.fill_(0)
        self.regression.output.bias.data.fill_(0)
        # 产生所有的预设置的anchor，default=3,4,5,6,7
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

    def forward(self,inputs):
        img_batch = inputs
        anchors = self.anchors(img_batch)
        
        C3, C4, C5 = self.resnet(inputs)
        fpn_out_5_layer_list = self.fpn([C3, C4, C5])
        # 沿着列进行整合
        regression = torch.cat([self.regression(feature) for feature in fpn_out_5_layer_list], dim=1)
        classification = torch.cat([self.classification(feature) for feature in fpn_out_5_layer_list], dim=1)

        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        # 这里是我们最后要得到的anchor对应的类别（也就是id:0~79）![...]
        finalAnchorBoxesIndexes = torch.Tensor([]).long()
        # 最后我们要得到的coordinates 坐标,size == [...,4]
        finalAnchorBoxesCoordinates = torch.Tensor([])

        # if torch.cuda.is_available():
        #     finalScores = finalScores.cuda()
        #     finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
        #     finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

        # 试一下可不可以运行，不自己设置一个的话会一直continue
        classification = torch.randn([2, 49104, 80])
        # start = time.time()
        # 这个就是针对一个类的所有anchor的score进行处理
        for i in range(classification.shape[2]):
            # squeeze得到我们的scores.shape == [2,...]
            scores = torch.squeeze(classification[:, :, i])
            # scores是tensor,pytorch里面有重写这个>运算符，scores_over_thresh为[True,False,....]
            # 先初步筛选>0.05分数的框
            scores_over_thresh = (scores > 0.05)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue
            # 只把刚刚判定为True的部分提取出来，放入scores中
            scores = scores[scores_over_thresh]
            # 这个应该是不改变tensor的shape的吧
            anchorBoxes = torch.squeeze(transformed_anchors)
            # 把scores达标的anchor先挑出来，这里产生的anchorBoxes的size是[...,4]，坐标是x1y1x2y2
            anchorBoxes = anchorBoxes[scores_over_thresh]
            # nms 返回一个一维的tensor
            # scores是一维tensor，长度应该和anchorBoxes的一维长度一样的，对应每一个anchor的scores,最后一个是iou theshold是一个计算iou的参数
            anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
            # 上面anchors_nms_idx得到所有anchors的id(大家可以打印下或者debug里面看，anchors_nms_idx都是整数！！！！)
            # 所以这个finalScores就是利用idx找出对应anchor的scores!
            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            # 这里的finalAnchorBoxesIndexesValue就是变成[0,0,0,...,1,1,1,...,79,79]
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            # if torch.cuda.is_available():
            #     finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()
            # [0, 0, 0, ..., 1, 1, 1, ..., 79, 79]
            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            # final anchor是[...,4]也就是我们最后得到的要输出的anchor！！！
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        # end = time.time()
        # print(end-start) #my python nms :359.5818974971771 torch nms:91.24284791946411
        # 最终我们是得到了结果首先是每个anchors的score,然后是对应的类别数，最后是坐标
        # e.g
        # torch.Size([355672])
        # torch.Size([355672])
        # torch.Size([355672, 4])
        return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
        # return regression,classification,transformed_anchors
    

if __name__ == '__main__':
    
    C = torch.randn([2,3,512,512])
    model = RetinaNet(80)
    out = model(C)
    # print(out.shape)
    for i in range(len(out)):
        print(out[i].shape)
        print(out[i])

    # scores
    # torch.Size([355672])
    # tensor([4.4658, 4.3042, 4.2936, ..., 0.0545, 0.0525, 0.0511])
    # torch.Size([355672])
    # id
    # tensor([0, 0, 0, ..., 79, 79, 79])
    # torch.Size([355672, 4])
    # anchors
    # tensor([[143.4912, 469.7456, 200.5088, 498.2544],
    #         [39.4912, 373.7456, 96.5088, 402.2544],
    #         [456.6863, 229.3726, 479.3137, 274.6274],
    #         ...,
    #         [192.6863, 45.3726, 215.3137, 90.6274],
    #         [193.3726, 298.7452, 238.6274, 389.2548],
    #         [392.6863, 0.0000, 415.3137, 34.6274]], grad_fn= < CatBackward >)

    
"""if __name__ == '__main__':
    C = torch.randn([2,3,512,512])
    model = RetinaNet(80)
    out = model(C)
    for i in range(len(out)):
        print(out[i].shape)
        # torch.Size([2, 49104, 4])
        # torch.Size([2, 49104, 80])"""