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

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

from ..models import SUPPORTING_TASK_LIST
from ..models.utils import get_model_format

OUTPUT_ROOT_DIR = "./outputs"
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


@dataclass
class ConfigSummary:
    task: Optional[str] = None
    model_name: Optional[str] = None
    is_graphmodule_training: Optional[bool] = None
    logging_dir: Optional[Path] = None


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args_netspresso(with_gpus=False, isTrain=True):

    parser = argparse.ArgumentParser(description="Parser for NetsPresso configuration")

    # -------- User arguments ----------------------------------------

    if with_gpus:
        parser.add_argument(
            '--gpus', type=parse_gpu_ids, default="",
            dest='gpus',
            help='GPU device indices (comma-separated)')

    parser.add_argument(
        '--data', type=str, required=True,
        dest='data',
        help="Config for dataset information")

    parser.add_argument(
        '--augmentation', type=str, required=True,
        dest='augmentation',
        help="Config for data augmentation")

    parser.add_argument(
        '--model', type=str, required=True,
        dest='model',
        help="Config for the model architecture")

    if isTrain:
        parser.add_argument(
            '--training', type=str, required=True,
            dest='training',
            help="Config for training options")

    parser.add_argument(
        '--logging', type=str, default='config/logging.yaml',
        dest='logging',
        help="Config for logging options")

    parser.add_argument(
        '--environment', type=str, default='config/environment.yaml',
        dest='environment',
        help="Config for training environment (# workers, etc.)")

    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default=LOG_LEVEL,
        dest='log_level',
        help="Logging level in training process")

    parser.add_argument(
        '--task', type=str, default=None,
        dest='task',
        help="")

    parser.add_argument(
        '--model-name', type=str, default=None,
        dest='model_name',
        help="")

    parser.add_argument(
        '--is-graphmodule-training', type=str2bool, default=None,
        dest='is_graphmodule_training',
        help="")

    parser.add_argument(
        '--logging-dir', type=Path, default=None,
        dest='logging_dir',
        help="")

    args_parsed, _ = parser.parse_known_args()

    return args_parsed


def parse_gpu_ids(gpu_arg: str) -> Optional[Union[List, int]]:
    """Parse comma-separated GPU IDs and return as a list of integers."""
    if gpu_arg is None or str(gpu_arg) in ["", "None"]:
        return None

    try:
        gpu_ids = [int(id) for id in gpu_arg.split(',')]

        if len(gpu_ids) == 1:  # Single GPU
            return gpu_ids[0]

        gpu_ids = sorted(gpu_ids)
        return gpu_ids
    except ValueError as e:
        raise argparse.ArgumentTypeError('Invalid GPU IDs. Please provide comma-separated integers.') from e


def set_arguments(
    data: Union[Path, str],
    augmentation: Union[Path, str],
    model: Union[Path, str],
    logging: Union[Path, str],
    environment: Union[Path, str],
    training: Optional[Union[Path, str]] = None,
) -> DictConfig:

    conf_data = OmegaConf.load(data)
    conf_augmentation = OmegaConf.load(augmentation)
    conf_model = OmegaConf.load(model)
    if training:
        conf_training = OmegaConf.load(training)
    conf_logging = OmegaConf.load(logging)
    conf_environment = OmegaConf.load(environment)

    conf = OmegaConf.create()
    conf.merge_with(conf_data)
    conf.merge_with(conf_augmentation)
    conf.merge_with(conf_model)
    if training:
        conf.merge_with(conf_training)
    conf.merge_with(conf_logging)
    conf.merge_with(conf_environment)

    return conf


def get_gpu_from_config(conf_environment: DictConfig) -> Optional[Union[List, int]]:
    conf_environment_gpus = str(conf_environment.gpus) if hasattr(conf_environment, 'gpus') else None
    return parse_gpu_ids(conf_environment_gpus)


def get_gpus_from_parser_and_config(
    gpus: Optional[Union[List, int]],
    conf_environment: DictConfig
) -> Union[List, int]:
    conf_environment_gpus = get_gpu_from_config(conf_environment)
    if gpus is None:
        if conf_environment_gpus is None:
            return 0  # Try use the 'cuda:0'
        return conf_environment_gpus
    return gpus


def get_new_logging_dir(output_root_dir, project_id, mode: Literal['training', 'evaluation', 'inference']):
    version_idx = 0
    project_dir: Path = Path(output_root_dir) / project_id

    while (project_dir / f"version_{version_idx}").exists():
        version_idx += 1

    new_logging_dir: Path = project_dir / f"version_{version_idx}"
    new_logging_dir.mkdir(exist_ok=True, parents=True)

    summary_path = new_logging_dir / f"{mode}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({"status": "", "error_stats": ""}, f, indent=4)

    return new_logging_dir


def validate_train_config(conf: DictConfig) -> ConfigSummary:
    # Get information from configuration
    model_format = get_model_format(conf.model)
    assert model_format in ['torch', 'torch.fx'], "Training model format must be either 'torch' or 'torch.fx'. (.safetensors or .pt)"
    is_graphmodule_training = model_format == 'torch.fx'

    task = str(conf.model.task).lower()
    assert task in SUPPORTING_TASK_LIST

    model_name = str(conf.model.name).lower()

    if is_graphmodule_training:
        model_name += "_graphmodule"

    project_id = conf.logging.project_id if conf.logging.project_id is not None else f"{task}_{model_name}"
    logging_dir: Path = get_new_logging_dir(output_root_dir=conf.logging.output_dir, project_id=project_id, mode='training')

    return ConfigSummary(task=task, model_name=model_name, is_graphmodule_training=is_graphmodule_training, logging_dir=logging_dir)


def validate_evaluation_config(conf: DictConfig, gpus: Union[List, int]) -> ConfigSummary:

    task = str(conf.model.task).lower()
    assert task in SUPPORTING_TASK_LIST

    model_format = get_model_format(conf.model)
    if model_format == 'onnx':
        assert isinstance(gpus, int), "ONNX model evaluation must be done on a single GPU."

    model_name = str(conf.model.name).lower() + '_evaluation'

    project_id = conf.logging.project_id if conf.logging.project_id is not None else f"{task}_{model_name}"
    logging_dir: Path = get_new_logging_dir(output_root_dir=conf.logging.output_dir, project_id=project_id, mode='evaluation')

    return ConfigSummary(task=task, model_name=model_name, is_graphmodule_training=None, logging_dir=logging_dir)


def validate_inference_config(conf: DictConfig, gpus: Union[List, int]) -> ConfigSummary:

    task = str(conf.model.task).lower()
    assert task in SUPPORTING_TASK_LIST

    model_format = get_model_format(conf.model)
    if model_format == 'onnx':
        assert isinstance(gpus, int), "ONNX model inference must be done on a single GPU."

    model_name = str(conf.model.name).lower() + '_inference'

    project_id = conf.logging.project_id if conf.logging.project_id is not None else f"{task}_{model_name}"
    logging_dir: Path = get_new_logging_dir(output_root_dir=conf.logging.output_dir, project_id=project_id, mode='inference')

    return ConfigSummary(task=task, model_name=model_name, is_graphmodule_training=None, logging_dir=logging_dir)
