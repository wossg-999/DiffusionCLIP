import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/mvtecad'''

UNI_MEDICAL_CLS_NAMES = ['brain', 'liver', 'retinal']
UNI_MEDICAL_ROOT = os.path.join(DATA_ROOT, 'uni-medical')

class uni_medical_Dataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=UNI_MEDICAL_CLS_NAMES, aug_rate=0.2, root=UNI_MEDICAL_ROOT, training=True):
        super(uni_medical_Dataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
