import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/cvc-clinicdb'''
Camelyon16_CLS_NAMES = [
    'Camelyon16',
]
Camelyon16_ROOT = os.path.join(DATA_ROOT, 'Camelyon16')

class Camelyon16Dataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=Camelyon16_CLS_NAMES, aug_rate=0.0, root=Camelyon16_ROOT, training=True):
        super(Camelyon16Dataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
