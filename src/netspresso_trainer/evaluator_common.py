from pathlib import Path
from typing import Literal, Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataloaders import build_dataloader, build_dataset
from .models import SUPPORTING_TASK_LIST, build_model, is_single_task_model
from .pipelines import build_pipeline
from .utils.environment import set_device
from .utils.logger import add_file_handler, set_logger


def evaluation_common(
    conf: DictConfig,
    task: str,
    model_name: str,
    logging_dir: Path,
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
):
    distributed, world_size, rank, devices = set_device(conf.environment.seed)
    logger = set_logger(level=log_level, distributed=distributed)

    conf.distributed = distributed
    conf.world_size = world_size
    conf.rank = rank

    # Basic setup
    add_file_handler(logging_dir / "result.log", distributed=conf.distributed)

    if not distributed or dist.get_rank() == 0:
        logger.info(f"Task: {task} | Model: {model_name}")
        logger.info(f"Result will be saved at {logging_dir}")

    if conf.distributed and conf.rank != 0:
        torch.distributed.barrier()  # wait for rank 0 to download dataset

    single_task_model = is_single_task_model(conf.model)
    conf.model.single_task_model = single_task_model

    # Build dataloader
    _, _, test_dataset = build_dataset(conf.data, conf.augmentation, task, model_name, distributed=distributed)
    assert test_dataset is not None, "For evaluation, valid split of dataset must be provided."
    if not distributed or dist.get_rank() == 0:
        logger.info(f"Summary | Dataset: <{conf.data.name}> (with {conf.data.format} format)")
        logger.info(f"Summary | Validation dataset: {len(test_dataset)} sample(s)")

    if conf.distributed and conf.rank == 0:
        torch.distributed.barrier()

    eval_dataloader = build_dataloader(conf, task, model_name, dataset=test_dataset, phase='val')

    # Build model
    # TODO: Not implemented for various model types. Only support pytorch model now
    model = build_model(
        conf.model, task, test_dataset.num_classes,
        model_checkpoint=conf.model.checkpoint.path,
        use_pretrained=conf.model.checkpoint.use_pretrained,
        img_size=conf.augmentation.img_size
    )

    model = model.to(device=devices)
    if conf.distributed:
        model = DDP(model, device_ids=[devices], find_unused_parameters=True)  # TODO: find_unused_parameters should be false (for now, PIDNet has problem)

    # Build evaluation pipeline
    pipeline_type = 'evaluation'
    pipeline = build_pipeline(pipeline_type=pipeline_type,
                              conf=conf,
                              task=task,
                              model_name=model_name,
                              model=model,
                              devices=devices,
                              class_map=test_dataset.class_map,
                              logging_dir=logging_dir,
                              is_graphmodule_training=None, # TODO: Remove is_graphmodule_training ...
                              dataloaders={'test': eval_dataloader})

    try:
        # Start evaluation
        pipeline.evaluation()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
