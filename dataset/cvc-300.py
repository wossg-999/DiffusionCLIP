import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/mvtecad'''

CVC_300_CLS_NAMES = ['colon']
CVC_300_ROOT = os.path.join(DATA_ROOT, 'CVC-300')

class CVC_300_Dataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=CVC_300_CLS_NAMES, aug_rate=0.2, root=CVC_300_ROOT, training=True):
        super(CVC_300_Dataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
