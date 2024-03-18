import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

from netspresso_trainer.trainer_common import train_common

from .models import SUPPORTING_TASK_LIST
from .utils.engine_utils import set_arguments, get_gpus_from_parser_and_config
from .utils.engine_utils import LOG_LEVEL, OUTPUT_ROOT_DIR


@dataclass
class ConfigSummary:
    task: Optional[str] = None
    model_name: Optional[str] = None
    is_graphmodule_training: Optional[bool] = None
    logging_dir: Optional[Path] = None


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


def validate_config(conf: DictConfig) -> ConfigSummary:
    # Get information from configuration
    is_graphmodule_training = bool(conf.model.checkpoint.fx_model_path)

    task = str(conf.model.task).lower()
    assert task in SUPPORTING_TASK_LIST

    model_name = str(conf.model.name).lower()

    if is_graphmodule_training:
        model_name += "_graphmodule"

    project_id = conf.logging.project_id if conf.logging.project_id is not None else f"{task}_{model_name}"
    logging_dir: Path = get_new_logging_dir(output_root_dir=conf.logging.output_dir, project_id=project_id)

    return ConfigSummary(task=task, model_name=model_name, is_graphmodule_training=is_graphmodule_training, logging_dir=logging_dir)


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
