import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/cvc-clinicdb'''
BRAIN_CLS_NAMES = [
    'brain',
]
Brain_ROOT = os.path.join(DATA_ROOT, 'uni-medical')

class BrainDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=BRAIN_CLS_NAMES, aug_rate=0.0, root=Brain_ROOT, training=True):
        super(BrainDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
