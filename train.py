import argparse
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

from dataloaders import build_dataset, build_dataloader
from models import build_model
from pipelines import ClassificationPipeline, SegmentationPipeline, DetectionPipeline
from utils.environment import set_device
from utils.logger import set_logger


SUPPORT_TASK = ['classification', 'segmentation', 'detection']
logger = set_logger('train', level=os.getenv('LOG_LEVEL', 'INFO'))


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
        '--profile', action='store_true',
        help="Whether to use profile mode")

    args, _ = parser.parse_known_args()

    return args


def train():
    args_parsed = parse_args_netspresso()
    args_data = OmegaConf.load(args_parsed.data)
    args_augmentation = OmegaConf.load(args_parsed.augmentation)
    args_model = OmegaConf.load(args_parsed.model)
    args_training = OmegaConf.load(args_parsed.training)
    args_logging = OmegaConf.load(args_parsed.logging)
    args_environment = OmegaConf.load(args_parsed.environment)
    
    args = OmegaConf.load(args_parsed.config)
    args = OmegaConf.merge(args, args_data)
    args = OmegaConf.merge(args, args_augmentation)
    args = OmegaConf.merge(args, args_model)
    args = OmegaConf.merge(args, args_training)
    args = OmegaConf.merge(args, args_logging)
    args = OmegaConf.merge(args, args_environment)
    
    distributed, world_size, rank, devices = set_device(args)

    args.distributed = distributed
    args.world_size = world_size
    args.rank = rank

    task = str(args.train.task).lower()
    assert task in SUPPORT_TASK
    model_name = args.train.architecture.full \
        if args.train.architecture.full is not None \
        else args.train.architecture.backbone
    model_name = str(model_name).lower()

    if args.distributed and args.rank != 0:
        torch.distributed.barrier()  # wait for rank 0 to download dataset

    train_dataset, valid_dataset, test_dataset = build_dataset(args)

    if args.distributed and args.rank == 0:
        torch.distributed.barrier()

    model = build_model(args, train_dataset.num_classes)

    train_dataloader, eval_dataloader = \
        build_dataloader(args, task, model, train_dataset=train_dataset, eval_dataset=valid_dataset, profile=args_parsed.profile)

    model = model.to(device=devices)
    if args.distributed:
        model = DDP(model, device_ids=[devices], find_unused_parameters=True)  # TODO: find_unused_parameters should be false (for now, PIDNet has problem)

    if task == 'classification':
        trainer = ClassificationPipeline(args, task, model_name, model, devices,
                                         train_dataloader, eval_dataloader, train_dataset.class_map,
                                         profile=args_parsed.profile)
    elif task == 'segmentation':
        trainer = SegmentationPipeline(args, task, model_name, model, devices,
                                       train_dataloader, eval_dataloader, train_dataset.class_map,
                                       profile=args_parsed.profile)
        
    elif task == 'detection':
        trainer = DetectionPipeline(args, task, model_name, model, devices,
                                    train_dataloader, eval_dataloader, train_dataset.class_map,
                                    profile=args_parsed.profile)

    else:
        raise AssertionError(f"No such task! (task: {task})")

    trainer.set_train()
    trainer.train()


if __name__ == '__main__':
    train()
