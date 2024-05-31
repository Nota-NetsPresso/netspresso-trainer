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

from omegaconf import OmegaConf

from .registry import SCHEDULER_DICT


def build_scheduler(optimizer, training_conf):
    scheduler_conf = training_conf.scheduler
    scheduler_name = scheduler_conf.name
    num_epochs = training_conf.epochs

    assert scheduler_name in SCHEDULER_DICT, f"{scheduler_name} not in scheduler dict!"
    lr_scheduler = SCHEDULER_DICT[scheduler_name](optimizer, scheduler_conf, num_epochs)

    return lr_scheduler, num_epochs
