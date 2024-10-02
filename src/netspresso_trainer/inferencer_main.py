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

import os
import subprocess
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from omegaconf import OmegaConf

from netspresso_trainer.inferencer_common import inference_common
from netspresso_trainer.utils.engine_utils import (
    LOG_LEVEL,
    get_gpus_from_parser_and_config,
    parse_args_netspresso,
    set_arguments,
    validate_inference_config,
)


def inference_with_yaml_impl(gpus: Optional[Union[List, int]], data: Union[Path, str], augmentation: Union[Path, str],
                         model: Union[Path, str], logging: Union[Path, str], environment: Union[Path, str], log_level: str = LOG_LEVEL):
    conf_environment = OmegaConf.load(environment).environment
    gpus = get_gpus_from_parser_and_config(gpus, conf_environment)
    assert isinstance(gpus, (list, int))

    gpu_ids_str = ','.join(map(str, gpus)) if isinstance(gpus, list) else str(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
    torch.cuda.empty_cache()  # Reinitialize CUDA to apply the change

    conf = set_arguments(data=data,
                         augmentation=augmentation,
                         model=model,
                         logging=logging,
                         environment=environment)
    config_summary = validate_inference_config(conf, gpus)

    try:
        if isinstance(gpus, int):
            inference_common(
                conf,
                task=config_summary.task,
                model_name=config_summary.model_name,
                logging_dir=config_summary.logging_dir,
                log_level=log_level
            )
        else:
            raise NotImplementedError
        return config_summary.logging_dir
    except Exception as e:
        raise e


def inference_cli() -> None:
    args_parsed = parse_args_netspresso(with_gpus=True, isTrain=False)

    logging_dir: Path = inference_with_yaml_impl(
        gpus=args_parsed.gpus,
        data=args_parsed.data,
        augmentation=args_parsed.augmentation,
        model=args_parsed.model,
        logging=args_parsed.logging,
        environment=args_parsed.environment,
        log_level=args_parsed.log_level
    )

    return logging_dir
