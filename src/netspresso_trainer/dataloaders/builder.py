import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from .detection import detection_collate_fn
from .registry import CREATE_TRANSFORM, CUSTOM_DATASET, DATA_SAMPLER, HUGGINGFACE_DATASET
from .utils.loader import create_loader

logger = logging.getLogger("netspresso_trainer")

TRAIN_VALID_SPLIT_RATIO = 0.9


def build_dataset(conf_data, conf_augmentation, task: str, model_name: str):

    logger.info('-'*40)
    logger.info("Loading data...")

    task = conf_data.task

    assert task in CREATE_TRANSFORM, f"The given task `{task}` is not supported!"
    assert task in DATA_SAMPLER, f"Data sampler for {task} is not yet supported!"

    train_transform = CREATE_TRANSFORM[task](model_name, is_training=True)
    target_transform = CREATE_TRANSFORM[task](model_name, is_training=False)

    data_format = conf_data.format

    assert data_format in ['local', 'huggingface'], f"No such data format named {data_format} in {['local', 'huggingface']}!"

    if data_format == 'local':
        assert task in CUSTOM_DATASET, f"Local dataset for {task} is not yet supported!"
        data_sampler = DATA_SAMPLER[task](conf_data, train_valid_split_ratio=TRAIN_VALID_SPLIT_RATIO)

        train_samples, valid_samples, test_samples, misc = data_sampler.load_samples()
        idx_to_class = misc['idx_to_class'] if 'idx_to_class' in misc else None
        label_value_to_idx = misc['label_value_to_idx'] if 'label_value_to_idx' in misc else None
        test_with_label = misc['test_with_label'] if 'test_with_label' in misc else None

        train_dataset = CUSTOM_DATASET[task](
            conf_data, conf_augmentation, model_name, idx_to_class=idx_to_class, split='train',
            samples=train_samples, transform=train_transform, label_value_to_idx=label_value_to_idx
        )

        valid_dataset = None
        if valid_samples is not None:
            valid_dataset = CUSTOM_DATASET[task](
                conf_data, conf_augmentation, model_name, idx_to_class=idx_to_class, split='valid',
                samples=valid_samples, transform=target_transform, label_value_to_idx=label_value_to_idx
            )

        test_dataset = None
        if test_samples is not None:
            test_dataset = CUSTOM_DATASET[task](
                conf_data, conf_augmentation, model_name, idx_to_class=idx_to_class, split='test',
                samples=test_samples, transform=target_transform,
                with_label=test_with_label, label_value_to_idx=label_value_to_idx
            )

    elif data_format == 'huggingface':
        assert task in CUSTOM_DATASET, f"HuggingFace dataset for {task} is not yet supported!"
        assert task in DATA_SAMPLER, f"Data sampler for {task} is not yet supported!"

        data_sampler = DATA_SAMPLER[task](conf_data, train_valid_split_ratio=TRAIN_VALID_SPLIT_RATIO)

        train_samples, valid_samples, test_samples, misc = data_sampler.load_huggingface_samples()
        idx_to_class = misc['idx_to_class'] if 'idx_to_class' in misc else None
        label_value_to_idx = misc['label_value_to_idx'] if 'label_value_to_idx' in misc else None
        test_with_label = misc['test_with_label'] if 'test_with_label' in misc else None

        train_dataset = HUGGINGFACE_DATASET[task](
            conf_data, conf_augmentation, model_name, idx_to_class=idx_to_class, split='train',
            huggingface_dataset=train_samples, transform=train_transform, label_value_to_idx=label_value_to_idx
        )

        valid_dataset = None
        if valid_samples is not None:
            valid_dataset = HUGGINGFACE_DATASET[task](
                conf_data, conf_augmentation, model_name, idx_to_class=idx_to_class, split='valid',
                huggingface_dataset=valid_samples, transform=target_transform, label_value_to_idx=label_value_to_idx
            )

        test_dataset = None
        if test_samples is not None:
            test_dataset = HUGGINGFACE_DATASET[task](
                conf_data, conf_augmentation, model_name, idx_to_class=idx_to_class, split='test',
                huggingface_dataset=test_samples, transform=target_transform, label_value_to_idx=label_value_to_idx
            )

    logger.info(f"Summary | Dataset: <{conf_data.name}> (with {data_format} format)")
    logger.info(f"Summary | Training dataset: {len(train_dataset)} sample(s)")
    if valid_dataset is not None:
        logger.info(f"Summary | Validation dataset: {len(valid_dataset)} sample(s)")
    if test_dataset is not None:
        logger.info(f"Summary | Test dataset: {len(test_dataset)} sample(s)")

    return train_dataset, valid_dataset, test_dataset


def build_dataloader(conf, task: str, model_name: str, train_dataset, eval_dataset, profile=False):

    if task == 'classification':
        collate_fn = None

        train_loader = create_loader(
            train_dataset,
            conf.data.name,
            logger,
            input_size=conf.augmentation.img_size,
            batch_size=conf.training.batch_size,
            is_training=True,
            num_workers=conf.environment.num_workers if not profile else 1,
            distributed=conf.distributed,
            collate_fn=collate_fn,
            pin_memory=False,
            world_size=conf.world_size,
            rank=conf.rank,
            kwargs=None
        )

        eval_loader = create_loader(
            eval_dataset,
            conf.data.name,
            logger,
            input_size=conf.augmentation.img_size,
            batch_size=conf.training.batch_size,
            is_training=False,
            num_workers=conf.environment.num_workers if not profile else 1,
            distributed=conf.distributed,
            collate_fn=None,
            pin_memory=False,
            world_size=conf.world_size,
            rank=conf.rank,
            kwargs=None
        )
    elif task == 'segmentation':
        collate_fn = None

        train_loader = create_loader(
            train_dataset,
            conf.data.name,
            logger,
            batch_size=conf.training.batch_size,
            is_training=True,
            num_workers=conf.environment.num_workers if not profile else 1,
            distributed=conf.distributed,
            collate_fn=collate_fn,
            pin_memory=False,
            world_size=conf.world_size,
            rank=conf.rank,
            kwargs=None
        )

        eval_loader = create_loader(
            eval_dataset,
            conf.data.name,
            logger,
            batch_size=conf.training.batch_size if model_name == 'pidnet' and not conf.distributed else 1,
            is_training=False,
            num_workers=conf.environment.num_workers if not profile else 1,
            distributed=conf.distributed,
            collate_fn=None,
            pin_memory=False,
            world_size=conf.world_size,
            rank=conf.rank,
            kwargs=None
        )
    elif task == 'detection':
        collate_fn = detection_collate_fn

        train_loader = create_loader(
            train_dataset,
            conf.data.name,
            logger,
            batch_size=conf.training.batch_size,
            is_training=True,
            num_workers=conf.environment.num_workers if not profile else 1,
            distributed=conf.distributed,
            collate_fn=collate_fn,
            pin_memory=False,
            world_size=conf.world_size,
            rank=conf.rank,
            kwargs=None
        )

        eval_loader = create_loader(
            eval_dataset,
            conf.data.name,
            logger,
            # TODO: support batch size 1 inference
            batch_size=conf.training.batch_size if not conf.distributed else 2,
            is_training=False,
            num_workers=conf.environment.num_workers if not profile else 1,
            distributed=conf.distributed,
            collate_fn=collate_fn,
            pin_memory=False,
            world_size=conf.world_size,
            rank=conf.rank,
            kwargs=None
        )

    else:
        raise AssertionError(f"Task ({task}) is not understood!")

    return train_loader, eval_loader
