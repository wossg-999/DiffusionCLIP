import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/mvtecad'''

KVASIR_CLS_NAMES = ['colon']
KVASIR_ROOT = os.path.join(DATA_ROOT, 'Kvasir/colon')

class kvasirDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=KVASIR_CLS_NAMES, aug_rate=0.2, root=KVASIR_ROOT, training=True):
        super(kvasirDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
