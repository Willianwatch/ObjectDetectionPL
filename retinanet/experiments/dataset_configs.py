from torchvision import transforms

from retinanet.materials.datasets.utils import Normalizer, Augmenter, Resizer


class CocoConfigs:
    def __init__(self):
        self.coco_path = ""
        self.train_set_name = "train2017"
        self.val_set_name = "val2017"
        self.train_transform = transforms.Compose([
            Normalizer(),
            Augmenter(),
            Resizer()
        ])
        self.val_transform = transforms.Compose([
            Normalizer(),
            Resizer()
        ])
        
           
class CSVConfigs:
    def __init__(self):
        self.csv_train = ""
        self.csv_val = ""
        self.csv_classes = []
        self.train_transforms = transforms.Compose([
            Normalizer(), 
            Augmenter(), 
            Resizer()
        ])
        self.val_transforms = transforms.Compose([
            Normalizer(), 
            Resizer()
        ])