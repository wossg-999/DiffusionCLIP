import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/mvtecad'''

COVID_CLS_NAMES = ['chest']
COVID_ROOT = os.path.join(DATA_ROOT, 'COVID-19_Radiography_Dataset')

class CovidDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=COVID_CLS_NAMES, aug_rate=0.2, root=COVID_ROOT, training=True):
        super(CovidDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
