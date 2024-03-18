import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

from .models import SUPPORTING_TASK_LIST
from .evaluator_common import evaluation_common
from .utils.engine_utils import ConfigSummary
from .utils.engine_utils import set_arguments, get_gpus_from_parser_and_config, get_new_logging_dir
from .utils.engine_utils import LOG_LEVEL, OUTPUT_ROOT_DIR


def validate_config(conf: DictConfig) -> ConfigSummary:

    task = str(conf.model.task).lower()
    assert task in SUPPORTING_TASK_LIST

    model_name = str(conf.model.name).lower() + '_evaluation'

    project_id = conf.logging.project_id if conf.logging.project_id is not None else f"{task}_{model_name}"
    logging_dir: Path = get_new_logging_dir(output_root_dir=conf.logging.output_dir, project_id=project_id, mode='evaluation')

    return ConfigSummary(task=task, model_name=model_name, is_graphmodule_training=None, logging_dir=logging_dir)


def evaluation_with_yaml_impl(gpus: Optional[Union[List, int]], data: Union[Path, str], augmentation: Union[Path, str],
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
    config_summary = validate_config(conf)

    try:
        if isinstance(gpus, int):
            evaluation_common(
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
