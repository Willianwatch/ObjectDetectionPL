import pytorch_lightning as pl
import torch
from torch import optim
from torchvision.ops import nms

from retinanet.methods.networks.nets.retinanet import resnet18
from retinanet.methods.utils import Anchors, ClipBoxes, BBoxTransform
from retinanet.methods.losses.focalloss import FocalLoss


class LitRetinaNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=80)
        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = FocalLoss()
        
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img_batch, annotations = batch
        classification, regression = self.forward(img_batch)
        anchors = self.anchors(img_batch)
        return self.focalLoss(classification, regression, anchors, annotations)
    
    def validation_step(self, batch, batch_idx):
        img_batch, annotations = batch
        classification, regression = self.forward(img_batch)
        anchors = self.anchors(img_batch)
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        finalResult = [[], [], []]

        finalScores = torch.Tensor([])
        finalAnchorBoxesIndexes = torch.Tensor([]).long()
        finalAnchorBoxesCoordinates = torch.Tensor([])

        if torch.cuda.is_available():
            finalScores = finalScores.cuda()
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

        for i in range(classification.shape[2]):
            scores = torch.squeeze(classification[:, :, i])
            scores_over_thresh = (scores > 0.05)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            if torch.cuda.is_available():
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)        
        return {
            "optimizer" : optimizer,
            "lr_scheduler" : scheduler,
            "monitor" : "train_loss"
        }
    