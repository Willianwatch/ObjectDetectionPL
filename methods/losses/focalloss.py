import torch.nn as nn
import torch


# 计算iou
# a ,b 格式为 x1,y1,x2,y2
def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    # 标签coco是xywh但是在加载的时候好像转化成了xyxy
    # anno torch.Size([1, 9, 5]) xyxy catagory
    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        # anchor is xyxy,so change it to xywh
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        
        # 最外面的都是batchsize
        for j in range(batch_size):
            # 取出对应的batch
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            # print(bbox_annotation.shape) [15, 5]
            # print(bbox_annotation)
            # 筛选 ！= -1
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            # print(bbox_annotation)
            # print(bbox_annotation.shape) [15, 5]

            # print(classification)
            # print(classification.shape)
            # clamp到0.0001~0.9999,其实这里用sigmoid也可以
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            # print(classification)
            # print(classification.shape)
            # 无GT
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    # append就是添加
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue
            
            # 计算iou
            # IoU [4567,15] 将所有的anchor和标签那15个anchor进行计算iou
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
            # max 4567,argmax 4567 就是坐标（index），选出15列每行(dim=1)最大的
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # compute the loss for classification
            # [4567,80] all is -1,device cpu
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
              targets = targets.cuda()

            # 全部设置为0
            targets[torch.lt(IoU_max, 0.4), :] = 0
            # 大于0.5的部分都设置为positive return [f,t,f,t,t,f,...]
            positive_indices = torch.ge(IoU_max, 0.5)

            # 算一下个数
            num_positive_anchors = positive_indices.sum()
            # [4567,5]
            assigned_annotations = bbox_annotation[IoU_argmax, :]

            # positive 部分 设置为1
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha
                
            # targets == 1的 用0.25,不是用0.75
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            # focal weight = a * w**gamma
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            # 交叉熵损失函数的计算[4567,80]
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                # 如果不等于-1
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                # [4567,80]
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
            # 把所有的都添加进去
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
            # compute the loss for regression

            if positive_indices.sum() > 0:
                # 把positive都找出来,33
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                # 计算gt xyxy -> xywh
                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                # smooth l1 loss
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                # Concatenates a sequence of tensors along a new dimension.
                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])   
                    
                # negative_indices = 1 + (~positive_indices)
                # smooth l1 loss
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                # 小于和大于0.111的用不同方式计算
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
        # 全部stack后平均，饭后最后的cls loss ,reg loss
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)     
    
    
if __name__ == '__main__':
    # def forward(self, classifications, regressions, anchors, annotations):
    c = torch.randn([1,4567,80]).cuda()
    r = torch.randn([1,4567,4]).cuda()
    a = torch.randn([1,4567,4]).cuda()
    anno = torch.randn([1,15,5]).cuda()
    model = FocalLoss().cuda()
    out = model(c,r,a,anno)
    # for i in range(len(out)):
    #     print(out[i])