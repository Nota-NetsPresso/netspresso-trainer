# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from typing import Callable, Dict

from .custom.image_proc import (
    AutoAugment,
    CenterCrop,
    ColorJitter,
    HSVJitter,
    Normalize,
    Pad,
    PoseTopDownAffine,
    RandomCrop,
    RandomErasing,
    RandomHorizontalFlip,
    RandomIoUCrop,
    RandomResize,
    RandomResizedCrop,
    RandomVerticalFlip,
    RandomZoomOut,
    Resize,
    ToTensor,
    TrivialAugmentWide,
)
from .custom.mixing import Mixing
from .custom.mosaic import MosaicDetection

TRANSFORM_DICT: Dict[str, Callable] = {
    'centercrop': CenterCrop,
    'colorjitter': ColorJitter,
    'pad': Pad,
    'randomcrop': RandomCrop,
    'randomresizedcrop': RandomResizedCrop,
    'randomhorizontalflip': RandomHorizontalFlip,
    'randomresize': RandomResize,
    'randomverticalflip': RandomVerticalFlip,
    'randomerasing': RandomErasing,
    'randomioucrop': RandomIoUCrop,
    'randomzoomout': RandomZoomOut,
    'resize': Resize,
    'mixing': Mixing,
    'mosaicdetection': MosaicDetection,
    'trivialaugmentwide': TrivialAugmentWide,
    'autoaugment': AutoAugment,
    'hsvjitter': HSVJitter,
    'posetopdownaffine': PoseTopDownAffine,
    'totensor': ToTensor,
    'normalize': Normalize,
}
