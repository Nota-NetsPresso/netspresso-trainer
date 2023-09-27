import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataloaders import build_dataloader, build_dataset
from .models import SUPPORTING_TASK_LIST, build_model, is_single_task_model
from .pipelines import build_pipeline
from .utils.environment import set_device
from .utils.logger import set_logger

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


def parse_args_netspresso():

    parser = argparse.ArgumentParser(description="Parser for NetsPresso configuration")

    # -------- User arguments ----------------------------------------

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
        help="Config for training environment (# workers, etc.)")

    args_parsed, _ = parser.parse_known_args()

    return args_parsed


def set_arguments(args_parsed):
    conf_data = OmegaConf.load(args_parsed.data)
    conf_augmentation = OmegaConf.load(args_parsed.augmentation)
    conf_model = OmegaConf.load(args_parsed.model)
    conf_training = OmegaConf.load(args_parsed.training)
    conf_logging = OmegaConf.load(args_parsed.logging)
    conf_environment = OmegaConf.load(args_parsed.environment)

    conf = OmegaConf.create()
    conf.merge_with(conf_data)
    conf.merge_with(conf_augmentation)
    conf.merge_with(conf_model)
    conf.merge_with(conf_training)
    conf.merge_with(conf_logging)
    conf.merge_with(conf_environment)

    return conf


def trainer():
    args_parsed = parse_args_netspresso()
    conf = set_arguments(args_parsed)

    assert bool(conf.model.fx_model_checkpoint) != bool(conf.model.checkpoint)
    is_graphmodule_training = bool(conf.model.fx_model_checkpoint)

    distributed, world_size, rank, devices = set_device(conf.training.seed)
    logger = set_logger(logger_name="netspresso_trainer", level=args_parsed.log_level, distributed=distributed)

    conf.distributed = distributed
    conf.world_size = world_size
    conf.rank = rank

    task = str(conf.model.task).lower()
    assert task in SUPPORTING_TASK_LIST

    # TODO: Get model name from checkpoint
    single_task_model = is_single_task_model(conf.model)
    conf_model_sub = conf.model.architecture.full if single_task_model else conf.model.architecture.backbone
    conf.model.single_task_model = single_task_model

    model_name = str(conf_model_sub.name).lower()

    if is_graphmodule_training:
        model_name += "_graphmodule"

    logger.info(f"Task: {task} | Model: {model_name} | Training with torch.fx model? {is_graphmodule_training}")

    if conf.distributed and conf.rank != 0:
        torch.distributed.barrier()  # wait for rank 0 to download dataset

    train_dataset, valid_dataset, test_dataset = build_dataset(conf.data, conf.augmentation, task, model_name)

    if conf.distributed and conf.rank == 0:
        torch.distributed.barrier()

    train_dataloader, eval_dataloader = \
        build_dataloader(conf, task, model_name, train_dataset=train_dataset, eval_dataset=valid_dataset)

    if is_graphmodule_training:
        assert conf.model.fx_model_checkpoint is not None
        assert Path(conf.model.fx_model_checkpoint).exists()
        model = torch.load(conf.model.fx_model_checkpoint)
    else:
        model = build_model(conf.model, task, train_dataset.num_classes, conf.model.checkpoint, conf.augmentation.img_size)

    model = model.to(device=devices)
    if conf.distributed:
        model = DDP(model, device_ids=[devices], find_unused_parameters=True)  # TODO: find_unused_parameters should be false (for now, PIDNet has problem)

    trainer = build_pipeline(conf, task, model_name, model,
                             devices, train_dataloader, eval_dataloader,
                             class_map=train_dataset.class_map,
                             is_graphmodule_training=is_graphmodule_training)

    trainer.set_train()
    try:
        trainer.train()

        if test_dataset:
            trainer.inference(test_dataset)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
