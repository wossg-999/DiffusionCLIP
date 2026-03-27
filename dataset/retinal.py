import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/cvc-clinicdb'''
retinal_CLS_NAMES = [
    'retinal',
]
retinal_ROOT = os.path.join(DATA_ROOT, 'uni-medical_3')

class retinalDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=retinal_CLS_NAMES, aug_rate=0.0, root=retinal_ROOT, training=True):
        super(retinalDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
