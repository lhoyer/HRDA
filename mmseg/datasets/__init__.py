# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional datasets

from .acdc import ACDCDataset
from .bdd100k import BDD100KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .dg_dataset import DGDataset
from .gta import GTADataset
from .mapillary import MapillaryDataset
from .synthia import SynthiaDataset
from .uda_dataset import UDADataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'GTADataset',
    'SynthiaDataset',
    'UDADataset',
    'DGDataset',
    'ACDCDataset',
    'DarkZurichDataset',
    'BDD100KDataset',
    'MapillaryDataset',
]
