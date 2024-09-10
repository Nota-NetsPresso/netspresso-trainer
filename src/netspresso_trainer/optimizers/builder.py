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

from typing import Literal

import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig

from .registry import OPTIMIZER_DICT


def build_overwrited_dict(optimizer_conf):
    tmp_optimizer_conf = optimizer_conf.copy()
    del tmp_optimizer_conf['overwrite'] # Remove unnecessary field
    overwrited_config_dict = {
        'backbone': tmp_optimizer_conf.copy(),
        'neck': tmp_optimizer_conf.copy(),
        'head': tmp_optimizer_conf.copy(),
    }

    overwrite_config = optimizer_conf.overwrite
    for module_overwrite_config in overwrite_config:
        if not hasattr(module_overwrite_config, 'param_group'):
            raise ValueError(f"Overwrite config must have 'param_group' field: {module_overwrite_config}")

        param_group = module_overwrite_config.param_group

        if hasattr(module_overwrite_config, 'lr'):
            overwrited_config_dict[param_group].lr = module_overwrite_config.lr
        if hasattr(module_overwrite_config, 'weight_decay'):
            overwrited_config_dict[param_group].weight_decay = module_overwrite_config.weight_decay
        if hasattr(module_overwrite_config, 'no_bias_decay'):
            overwrited_config_dict[param_group].no_bias_decay = module_overwrite_config.no_bias_decay
        if hasattr(module_overwrite_config, 'no_norm_weight_decay'):
            overwrited_config_dict[param_group].no_norm_weight_decay = module_overwrite_config.no_norm_weight_decay

    return overwrited_config_dict


def separate_no_weights_decay(params: set, module_config: DictConfig):
    bias_params = set(filter(lambda s: 'bias' in s, params))
    norm_params = set(filter(lambda s: 'norm.weight' in s, params))
    no_decay_params = set()
    if module_config.no_bias_decay:
        no_decay_params |= bias_params
        params -= bias_params
    if module_config.no_norm_weight_decay:
        no_decay_params |= norm_params
        params -= norm_params
    return params, no_decay_params


def split_param_groups(model, overwrited_config_dict):
    backbone_params = set()
    neck_params = set()
    head_params = set()

    # Separate parameters by module
    named_params_dict = dict(model.named_parameters())
    for k in named_params_dict:
        if k.startswith('backbone'):
            backbone_params.add(k)
        elif k.startswith('neck'):
            neck_params.add(k)
        elif k.startswith('head'):
            head_params.add(k)
        else:
            raise ValueError(f"Unknown parameter group: {k}") # Only for NetsPresso Trainer defined model

    # Separate parameters by weight, bias, and norm
    backbone_params = separate_no_weights_decay(backbone_params, overwrited_config_dict['backbone'])
    neck_params = separate_no_weights_decay(neck_params, overwrited_config_dict['neck'])
    head_params = separate_no_weights_decay(head_params, overwrited_config_dict['head'])

    param_set_dict = {'backbone': backbone_params, 'neck': neck_params, 'head': head_params}

    # Build param_groups and param_configs
    param_groups = []
    param_opt_configs = []
    for param_group_key, module_overwrite_config in overwrited_config_dict.items():
        # Remove unnecessary fields
        del module_overwrite_config['no_bias_decay']
        del module_overwrite_config['no_norm_weight_decay']

        params, no_decay_params = param_set_dict[param_group_key]

        module_overwrite_config.group_name = param_group_key + '_group'
        if params: # Skip if no parameters, e.g. no neck in the model
            param_groups.append([named_params_dict[k] for k in params]) # Convert from key to parameters
            param_opt_configs.append(module_overwrite_config.copy())

        module_overwrite_config.group_name = param_group_key + '_no_decay_group'
        module_overwrite_config.weight_decay = 0.0
        if no_decay_params: # Skip if no parameters, e.g. use weight decay for all parameters
            param_groups.append([named_params_dict[k] for k in no_decay_params])
            param_opt_configs.append(module_overwrite_config.copy())

    return param_groups, param_opt_configs

def build_optimizer(
    model,
    single_task_model: bool,
    optimizer_conf: DictConfig,
):
    no_dist_model = model.module if hasattr(model, 'module') else model

    opt_name: Literal['sgd', 'adam', 'adamw', 'adamax', 'adadelta', 'adagrad', 'rmsprop'] = optimizer_conf.name.lower()
    assert opt_name in OPTIMIZER_DICT

    overwrite_config = optimizer_conf.overwrite
    if overwrite_config:
        assert not single_task_model, "Overwrite config only for non single task model"
        overwrited_config_dict = build_overwrited_dict(optimizer_conf)
        param_groups, param_opt_configs = split_param_groups(no_dist_model, overwrited_config_dict)

        base_param_group = param_groups.pop(0)
        base_opt_config = param_opt_configs.pop(0)
        optimizer = OPTIMIZER_DICT[opt_name](base_param_group, base_opt_config)
        optimizer.param_groups[0]['group_name'] = base_opt_config.group_name # Add group name manually

        for param_group, opt_config in zip(param_groups, param_opt_configs):
            del opt_config['name'] # Remove unnecessary field
            optimizer.add_param_group({'params': param_group, **opt_config})

    else:
        named_params_dict = dict(no_dist_model.named_parameters())

        params = set(named_params_dict.keys())
        params, no_decay_params = separate_no_weights_decay(params, optimizer_conf)

        params = [named_params_dict[k] for k in params]
        no_decay_params = [named_params_dict[k] for k in no_decay_params]

        optimizer = OPTIMIZER_DICT[opt_name](params, optimizer_conf)
        if no_decay_params:
            optimizer.add_param_group({'params': no_decay_params, 'weight_decay': 0.0})

    return optimizer
