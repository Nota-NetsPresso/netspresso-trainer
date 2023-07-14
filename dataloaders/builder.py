import os
from pathlib import Path
import logging
from typing import List, Dict, Union, Optional, Type

from torch.utils.data import DataLoader

from dataloaders.base import BaseCustomDataset, BaseHFDataset

from dataloaders.classification import (
    ClassificationCustomDataset, ClassificationHFDataset, create_classification_transform
)
from dataloaders.segmentation import (
    SegmentationCustomDataset, SegmentationHFDataset, create_segmentation_transform
)
from dataloaders.detection import (
    DetectionCustomDataset, create_detection_transform, detection_collate_fn
)
from dataloaders.utils.loader import create_loader
from utils.logger import set_logger


_logger = set_logger('dataloaders', level=os.getenv('LOG_LEVEL', 'INFO'))

TRANSFORMER_COMPOSER = {
    'classification': create_classification_transform,
    'segmentation': create_segmentation_transform,
    'detection': create_detection_transform
}

CUSTOM_DATASET: Dict[str, Type[BaseCustomDataset]] = {
    'classification': ClassificationCustomDataset,
    'segmentation': SegmentationCustomDataset,
    'detection': DetectionCustomDataset
}

HUGGINGFACE_DATASET: Dict[str, Type[BaseHFDataset]] = {
    'classification': ClassificationHFDataset,
    'segmentation': SegmentationHFDataset
}


def build_dataset(args):

    _logger.info('-'*40)
    _logger.info('==> Loading data...')

    task = args.data.task

    assert task in TRANSFORMER_COMPOSER, f"The given task `{task}` is not supported!"

    train_transform = TRANSFORMER_COMPOSER[task](args, is_training=True)
    target_transform = TRANSFORMER_COMPOSER[task](args, is_training=False)
    
    data_format = args.data.format
    
    assert data_format in ['local', 'huggingface'], f"No such data format named {data_format} in {['local', 'huggingface']}!"
    
    if data_format == 'local':
        assert task in CUSTOM_DATASET, f"Local dataset for {task} is not yet supported!"
        
        train_samples, valid_samples, test_samples, misc = load_samples(args.data)
        idx_to_class = misc['idx_to_class'] if 'idx_to_class' in misc else None
        test_with_label = misc['test_with_label'] if 'test_with_label' in misc else None
        
        train_dataset = CUSTOM_DATASET[task](
            args, idx_to_class=idx_to_class, split='train',
            samples=train_samples, transform=train_transform
        )
        
        valid_dataset = None
        if valid_samples is not None:
            valid_dataset = CUSTOM_DATASET[task](
                args, idx_to_class=idx_to_class, split='valid',
                samples=valid_samples, transform=target_transform
            )
        
        test_dataset = None
        if test_samples is not None:
            test_dataset = CUSTOM_DATASET[task](
                args, idx_to_class=idx_to_class, split='test',
                samples=test_samples, transform=target_transform,
                with_label=test_with_label
            )
            
    elif data_format == 'huggingface':
        assert task in CUSTOM_DATASET, f"HuggingFace dataset for {task} is not yet supported!"
        
        train_samples, valid_samples, test_samples, misc = load_samples_huggingface(args.data)
        idx_to_class = misc['idx_to_class'] if 'idx_to_class' in misc else None
        
        train_dataset = HUGGINGFACE_DATASET[task](
            args, idx_to_class=idx_to_class, split='train',
            huggingface_dataset=train_samples, transform=train_transform
        )
        
        valid_dataset = None
        if valid_samples is not None:
            valid_dataset = HUGGINGFACE_DATASET[task](
                args, idx_to_class=idx_to_class, split='valid',
                huggingface_dataset=valid_samples, transform=target_transform
            )
        
        test_dataset = None
        if test_samples is not None:
            test_dataset = HUGGINGFACE_DATASET[task](
                args, idx_to_class=idx_to_class, split='test',
                huggingface_dataset=test_samples, transform=target_transform
            )

    _logger.info(f'Summary | Training dataset: {len(train_dataset)} sample(s)')
    if valid_dataset is not None:
        _logger.info(f'Summary | Validation dataset: {len(valid_dataset)} sample(s)')
    if test_dataset is not None:
        _logger.info(f'Summary | Test dataset: {len(test_dataset)} sample(s)')

    return train_dataset, valid_dataset, test_dataset


def build_dataloader(args, task, model, train_dataset, eval_dataset, profile):

    if task == 'classification':
        collate_fn = None

        train_loader = create_loader(
            train_dataset,
            args.data.name,
            _logger,
            input_size=args.training.img_size,
            batch_size=args.training.batch_size,
            is_training=True,
            num_workers=args.environment.num_workers if not profile else 1,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=False,
            world_size=args.world_size,
            rank=args.rank,
            kwargs=None
        )

        eval_loader = create_loader(
            eval_dataset,
            args.data.name,
            _logger,
            input_size=args.training.img_size,
            batch_size=args.training.batch_size,
            is_training=False,
            num_workers=args.environment.num_workers if not profile else 1,
            distributed=args.distributed,
            collate_fn=None,
            pin_memory=False,
            world_size=args.world_size,
            rank=args.rank,
            kwargs=None
        )
    elif task == 'segmentation':
        collate_fn = None

        train_loader = create_loader(
            train_dataset,
            args.data.name,
            _logger,
            batch_size=args.training.batch_size,
            is_training=True,
            num_workers=args.environment.num_workers if not profile else 1,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=False,
            world_size=args.world_size,
            rank=args.rank,
            kwargs=None
        )

        eval_loader = create_loader(
            eval_dataset,
            args.data.name,
            _logger,
            batch_size=args.training.batch_size if model == 'pidnet' and not args.distributed else 1,
            is_training=False,
            num_workers=args.environment.num_workers if not profile else 1,
            distributed=args.distributed,
            collate_fn=None,
            pin_memory=False,
            world_size=args.world_size,
            rank=args.rank,
            kwargs=None
        )
    elif task == 'detection':
        collate_fn = detection_collate_fn

        train_loader = create_loader(
            train_dataset,
            args.data.name,
            _logger,
            batch_size=args.training.batch_size,
            is_training=True,
            num_workers=args.environment.num_workers if not profile else 1,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=False,
            world_size=args.world_size,
            rank=args.rank,
            kwargs=None
        )

        eval_loader = create_loader(
            eval_dataset,
            args.data.name,
            _logger,
            # TODO: support batch size 1 inference
            batch_size=args.training.batch_size if not args.distributed else 2,
            is_training=False,
            num_workers=args.environment.num_workers if not profile else 1,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=False,
            world_size=args.world_size,
            rank=args.rank,
            kwargs=None
        )

        # train_loader = DataLoader(train_dataset, batch_size=args.training.batch_size,
        #                           num_workers=args.environment.num_workers if not profile else 1,
        #                           shuffle=True,
        #                           collate_fn=None,
        #                           pin_memory=False)
        # eval_loader = DataLoader(eval_dataset, batch_size=1,
        #                          num_workers=args.environment.num_workers if not profile else 1,
        #                          shuffle=False,
        #                          collate_fn=None,
        #                          pin_memory=False)
    else:
        raise AssertionError(f"Task ({task}) is not understood!")

    return train_loader, eval_loader
