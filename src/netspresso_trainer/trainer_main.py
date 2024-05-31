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

from netspresso_trainer.trainer_common import train_common
from netspresso_trainer.utils.engine_utils import (
    LOG_LEVEL,
    get_gpus_from_parser_and_config,
    parse_args_netspresso,
    parse_gpu_ids,
    set_arguments,
    validate_train_config,
)


def run_distributed_training_script(gpu_ids, data, augmentation, model, training, logging, environment, log_level,
                                    task, model_name, is_graphmodule_training, logging_dir):

    command = [
        "--data", data,
        "--augmentation", augmentation,
        "--model", model,
        "--training", training,
        "--logging", logging,
        "--environment", environment,
        "--log-level", log_level,
        "--task", task,
        "--model-name", model_name,
        "--is-graphmodule-training", is_graphmodule_training,
        "--logging-dir", logging_dir,
    ]

    # Distributed training script
    command = [
        'python', '-m', 'torch.distributed.launch',
        f'--nproc_per_node={len(gpu_ids)}',  # GPU #
        f"{Path(__file__).absolute().parent / 'trainer_main.py'}", *map(str, command)
    ]

    # Run subprocess
    process = subprocess.Popen(command)

    try:
        process.wait()
    except KeyboardInterrupt:
        print("Interrupted. Terminating the training process...")
        process.terminate()
        process.wait()


def train_with_yaml_impl(gpus: Optional[Union[List, int]], data: Union[Path, str], augmentation: Union[Path, str],
                         model: Union[Path, str], training: Union[Path, str],
                         logging: Union[Path, str], environment: Union[Path, str], log_level: str = LOG_LEVEL):
    conf_environment = OmegaConf.load(environment).environment
    gpus = get_gpus_from_parser_and_config(gpus, conf_environment)
    assert isinstance(gpus, (list, int))

    gpu_ids_str = ','.join(map(str, gpus)) if isinstance(gpus, list) else str(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
    torch.cuda.empty_cache()  # Reinitialize CUDA to apply the change

    conf = set_arguments(data=data,
                         augmentation=augmentation,
                         model=model,
                         training=training,
                         logging=logging,
                         environment=environment)
    config_summary = validate_train_config(conf)

    try:
        if isinstance(gpus, int):
            train_common(
                conf,
                task=config_summary.task,
                model_name=config_summary.model_name,
                is_graphmodule_training=config_summary.is_graphmodule_training,
                logging_dir=config_summary.logging_dir,
                log_level=log_level
            )
        else:
            run_distributed_training_script(
                gpus, data, augmentation, model, training, logging, environment, log_level,
                config_summary.task, config_summary.model_name, config_summary.is_graphmodule_training, config_summary.logging_dir
            )
        return config_summary.logging_dir
    except Exception as e:
        raise e


def train_with_yaml(
    data: Union[Path, str],
    augmentation: Union[Path, str],
    model: Union[Path, str], training: Union[Path, str],
    logging: Union[Path, str], environment: Union[Path, str],
    gpus: Optional[str] = None, log_level: str = LOG_LEVEL
):

    gpus: Union[List, int] = parse_gpu_ids(gpus)

    logging_dir: Path = train_with_yaml_impl(
        gpus=gpus,
        data=data,
        augmentation=augmentation,
        model=model,
        training=training,
        logging=logging,
        environment=environment,
        log_level=log_level
    )

    return logging_dir


def train_cli() -> None:
    args_parsed = parse_args_netspresso(with_gpus=True, isTrain=True)

    logging_dir: Path = train_with_yaml_impl(
        gpus=args_parsed.gpus,
        data=args_parsed.data,
        augmentation=args_parsed.augmentation,
        model=args_parsed.model,
        training=args_parsed.training,
        logging=args_parsed.logging,
        environment=args_parsed.environment,
        log_level=args_parsed.log_level
    )

    return logging_dir


def train_cli_without_additional_gpu_check() -> None:
    args_parsed = parse_args_netspresso(with_gpus=False, isTrain=True)

    conf = set_arguments(
        data=args_parsed.data,
        augmentation=args_parsed.augmentation,
        model=args_parsed.model,
        training=args_parsed.training,
        logging=args_parsed.logging,
        environment=args_parsed.environment
    )

    train_common(
        conf,
        task=args_parsed.task,
        model_name=args_parsed.model_name,
        is_graphmodule_training=args_parsed.is_graphmodule_training,
        logging_dir=args_parsed.logging_dir,
        log_level=args_parsed.log_level
    )

if __name__ == "__main__":

    # Execute by `run_distributed_training_script`
    train_cli_without_additional_gpu_check()
