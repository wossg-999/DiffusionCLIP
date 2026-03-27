import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/cvc-clinicdb'''
LIVER_CLS_NAMES = [
    'liver',
]
Liver_ROOT = os.path.join(DATA_ROOT, 'uni-medical_2')

class liverDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=LIVER_CLS_NAMES, aug_rate=0.0, root=Liver_ROOT, training=True):
        super(liverDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
