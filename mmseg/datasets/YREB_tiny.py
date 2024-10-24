# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio
import os.path as osp
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset, BaseCDDataset
from typing import Dict, List, Optional, Sequence, Union
import mmengine
import logging
logger = logging.getLogger(__name__)
import pdb

from mmseg.datasets import BaseCDDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class YREBtinyDataset(BaseCDDataset):
    """Whispers'2024 challenge dataset.
    
    """
    
    METAINFO = dict(
    classes=(
        'Tree', 
        'Grassland', 
        'Cropland', 
        'Low Vegetation', 
        'Wetland', 
        'Water', 
        'Built-up', 
        'Bare ground', 
        'Snow'
    ),
    palette=[
        [0, 100, 0],      # Tree: 짙은 녹색
        [124, 252, 0],    # Grassland: 밝은 녹색
        [218, 165, 32],   # Cropland: 황금색
        [144, 238, 144],  # Low Vegetation: 연한 녹색
        [0, 128, 128],    # Wetland: 청록색
        [0, 0, 255],      # Water: 파란색
        [255, 0, 0],      # Built-up: 빨간색
        [165, 42, 42],    # Bare ground: 갈색
        [255, 255, 255],  # Snow: 흰색
    ]
    )

    
    def __init__(self,
                 img_suffix='.tif',
                 img_suffix2='.tif',
                 seg_map_suffix='.tif',
                 ignore_index=255,
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            img_suffix2=img_suffix2,
            ignore_index= ignore_index,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)