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

from typing import Dict, Type

import torch
from torch.optim.lr_scheduler import _LRScheduler

from .cosine_lr import CosineAnnealingLRWithCustomWarmUp
from .cosine_warm_restart import CosineAnnealingWarmRestartsWithCustomWarmUp
from .multi_step_lr import MultiStepLR
from .poly_lr import PolynomialLRWithWarmUp
from .step_lr import StepLR

SCHEDULER_DICT: Dict[str, Type[_LRScheduler]] = {
    'cosine': CosineAnnealingWarmRestartsWithCustomWarmUp,
    'cosine_no_sgdr': CosineAnnealingLRWithCustomWarmUp,
    'multi_step': MultiStepLR,
    'poly': PolynomialLRWithWarmUp,
    'step': StepLR
}
