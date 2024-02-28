import json
import os
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image as Image
import torch
from omegaconf import DictConfig
from torch.utils.data import random_split

from ..base import BaseDataSampler
from ..utils.constants import IMG_EXTENSIONS
from ..utils.misc import natural_key


class PoseEstimationDataSampler(BaseDataSampler):
    def __init__(self, conf_data, train_valid_split_ratio):
        super(PoseEstimationDataSampler, self).__init__(conf_data, train_valid_split_ratio)

    def load_data(self, split='train'):
        pass

    def load_samples(self):
        pass

    def load_huggingface_samples(self):
        raise NotImplementedError
