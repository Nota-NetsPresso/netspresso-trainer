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

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from .base import TrainingLogger


def build_logger(conf, task: str, model_name: str, step_per_epoch: int, class_map: Dict[int, str], num_sample_images: int, result_dir: Union[Path, str], epoch: Optional[int] = None):
    training_logger = TrainingLogger(conf,
                                     task=task, model=model_name,
                                     step_per_epoch=step_per_epoch,
                                     class_map=class_map, num_sample_images=num_sample_images,
                                     result_dir=result_dir,
                                     epoch=epoch)

    return training_logger
