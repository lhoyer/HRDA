# ---------------------------------------------------------------
# Copyright (c) 2023 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class MapillaryDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(MapillaryDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='_labelTrainIds.png', **kwargs)
        self.valid_mask_size = None
