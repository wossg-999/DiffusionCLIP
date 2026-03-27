import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/mvtecad'''

ENDO_CLS_NAMES = ['colon']
ENDO_ROOT = os.path.join(DATA_ROOT, 'EndoTect/endo')

class endoDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=ENDO_CLS_NAMES, aug_rate=0.2, root=ENDO_ROOT, training=True):
        super(endoDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
