import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/cvc-clinicdb'''
BraTS2021_CLS_NAMES = [
    'BraTS2021',
]
BraTS2021_ROOT = os.path.join(DATA_ROOT, 'BraTS2021')

class BraTS2021Dataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=BraTS2021_CLS_NAMES, aug_rate=0.0, root=BraTS2021_ROOT, training=True):
        super(BraTS2021Dataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
