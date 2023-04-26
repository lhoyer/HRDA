# ---------------------------------------------------------------
# Copyright (c) 2023 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.datasets.uda_dataset import UDADataset
from .builder import DATASETS


@DATASETS.register_module()
class DGDataset(UDADataset):

    def __init__(self, source, cfg):
        self.source = source
        self.CLASSES = source.CLASSES
        self.PALETTE = source.PALETTE
        self._setup_rcs(cfg)
        assert cfg.get('sync_crop_size') is None

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_source_sample()
        else:
            return self.source[idx]

    def __len__(self):
        return len(self.source)
