import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

from netspresso_trainer.cfg import TrainerConfig
from netspresso_trainer.trainer_common import train_common

from .models import SUPPORTING_TASK_LIST

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

def parse_args_netspresso(with_gpus=False):

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
        f"{Path(__file__).absolute().parent / 'trainer_cli_multi_gpu.py'}", *map(str, command)
    ]

    # Run subprocess
    process = subprocess.Popen(command)

    try:
        process.wait()
    except KeyboardInterrupt:
        print("Interrupted. Terminating the training process...")
        process.terminate()
        process.wait()


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


def get_new_logging_dir(output_root_dir, project_id, initialize=True):
    version_idx = 0
    project_dir: Path = Path(output_root_dir) / project_id

    while (project_dir / f"version_{version_idx}").exists():
        version_idx += 1

    new_logging_dir: Path = project_dir / f"version_{version_idx}"
    new_logging_dir.mkdir(exist_ok=True, parents=True)

    if initialize:
        summary_path = new_logging_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({"success": False}, f, indent=4)

    return new_logging_dir


def set_arguments(
    data: Union[Path, str],
    augmentation: Union[Path, str],
    model: Union[Path, str],
    training: Union[Path, str],
    logging: Union[Path, str],
    environment: Union[Path, str]
) -> DictConfig:

    conf_data = OmegaConf.load(data)
    conf_augmentation = OmegaConf.load(augmentation)
    conf_model = OmegaConf.load(model)
    conf_training = OmegaConf.load(training)
    conf_logging = OmegaConf.load(logging)
    conf_environment = OmegaConf.load(environment)

    conf = OmegaConf.create()
    conf.merge_with(conf_data)
    conf.merge_with(conf_augmentation)
    conf.merge_with(conf_model)
    conf.merge_with(conf_training)
    conf.merge_with(conf_logging)
    conf.merge_with(conf_environment)

    return conf

def validate_config(conf: DictConfig) -> ConfigSummary:
    # Get information from configuration
    assert bool(conf.model.fx_model_checkpoint) != bool(conf.model.checkpoint)
    is_graphmodule_training = bool(conf.model.fx_model_checkpoint)

    task = str(conf.model.task).lower()
    assert task in SUPPORTING_TASK_LIST

    model_name = str(conf.model.name).lower()

    if is_graphmodule_training:
        model_name += "_graphmodule"

    project_id = conf.logging.project_id if conf.logging.project_id is not None else f"{task}_{model_name}"
    logging_dir: Path = get_new_logging_dir(output_root_dir="./outputs", project_id=project_id)

    return ConfigSummary(task=task, model_name=model_name, is_graphmodule_training=is_graphmodule_training, logging_dir=logging_dir)

def set_struct_recursive(conf: DictConfig, value: bool) -> None:
    OmegaConf.set_struct(conf, value)

    for _, conf_value in conf.items():
        if isinstance(conf_value, DictConfig):
            set_struct_recursive(conf_value, value)

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


def train_with_yaml_impl(gpus: Optional[Union[List, int]], data: Union[Path, str], augmentation: Union[Path, str],
                         model: Union[Path, str], training: Union[Path, str],
                         logging: Union[Path, str], environment: Union[Path, str], log_level: str = LOG_LEVEL):
    conf_environment = OmegaConf.load(environment).environment
    gpus = get_gpus_from_parser_and_config(gpus, conf_environment)
    assert isinstance(gpus, (list, int))

    gpu_ids_str = ','.join(map(str, gpus)) if isinstance(gpus, list) else str(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
    torch.cuda.empty_cache()  # Reinitialize CUDA to apply the change

    conf = set_arguments(data, augmentation, model, training, logging, environment)
    config_summary = validate_config(conf)

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

def train_with_config_impl(gpus: int, config: TrainerConfig, log_level: str = LOG_LEVEL):

    gpus = get_gpus_from_parser_and_config(gpus, config.environment)
    assert isinstance(gpus, int), f"Currently, only single-GPU training is supported in this API. Your gpu(s): {gpus}"

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
    torch.cuda.empty_cache()  # Reinitialize CUDA to apply the change

    conf: DictConfig = OmegaConf.create(config)
    set_struct_recursive(conf, False)
    config_summary = validate_config(conf)

    try:
        train_common(
            conf,
            task=config_summary.task,
            model_name=config_summary.model_name,
            is_graphmodule_training=config_summary.is_graphmodule_training,
            logging_dir=config_summary.logging_dir,
            log_level=log_level
        )
        return config_summary.logging_dir
    except Exception as e:
        raise e
