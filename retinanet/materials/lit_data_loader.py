from typing import Optional

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from retinanet.experiments.dataset_configs import CocoConfigs, CSVConfigs
from retinanet.materials.datasets.coco_dataset import CocoDataset
from retinanet.materials.datasets.csv_dataset import CSVDataset
from retinanet.materials.datasets.utils import AspectRatioBasedSampler, collater


class LitDataLoader(pl.LightningDataModule):
    def __init__(self, dataset_name: str):
        super().__init__()
        if dataset_name in ["coco", "csv"]:
            self.dataset_name = dataset_name
        else:
            raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
            
    def setup(self, stage: Optional[str] = None):
        if self.dataset_name == "coco":
            coco_configs = CocoConfigs()
            self.dataset_train = CocoDataset(coco_configs.coco_path, set_name=coco_configs.train_set_name,
                                        transform=coco_configs.train_transform)
            self.dataset_val = CocoDataset(coco_configs.coco_path, set_name=coco_configs.val_set_name,
                                      transform=coco_configs.val_transform)
        else:
            csv_configs = CSVConfigs()
            self.dataset_train = CSVDataset(train_file=csv_configs.csv_train, class_list=csv_configs.csv_classes,
                                    transform=csv_configs.train_transforms)

            self.dataset_val = CSVDataset(train_file=csv_configs.csv_val, class_list=csv_configs.csv_classes,
                                     transform=csv_configs.val_transforms)
            
    def train_dataloader(self):
        sampler = AspectRatioBasedSampler(self.dataset_train, batch_size=2, drop_last=False)
        return DataLoader(self.dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)


    def val_dataloader(self):
        sampler_val = AspectRatioBasedSampler(self.dataset_val, batch_size=1, drop_last=False)
        return DataLoader(self.dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
        
    def test_dataloader(self):
        return super().test_dataloader()