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

import math
from copy import deepcopy

import torch

__all__ = ['ModelEMA', 'ExpDecayModelEMA']


class ModelEMA:
    def __init__(self, model, decay, device=None):
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_param = model_param.to(device=self.device)
                updated_param = self.decay * ema_param + (1. - self.decay) * model_param
                ema_param.copy_(updated_param)


class ExpDecayModelEMA:
    def __init__(self, model, decay, beta, device=None):
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.beta = beta
        self.device = device
        self.counter = 0
        if self.device is not None:
            self.module.to(device=device)

    def get_decay(self):
        return self.decay * (1 - math.exp(-self.counter / self.beta))

    def update(self, model):
        decay = self.get_decay()
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_param = model_param.to(device=self.device)
                updated_param = decay * ema_param + (1. - decay) * model_param
                ema_param.copy_(updated_param)
        self.counter += 1


def build_ema(model, conf):
    decay_type = conf.training.ema.name
    if decay_type == 'constant_decay':
        return ModelEMA(model=model, decay=conf.training.ema.decay)
    elif decay_type == 'exp_decay':
        return ExpDecayModelEMA(model=model, decay=conf.training.ema.decay, beta=conf.training.ema.beta)
    else:
        raise ValueError(f"Unsupported EMA decay type. training.ema_decay.name: {decay_type}")
