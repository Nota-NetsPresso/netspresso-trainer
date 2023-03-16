from pathlib import Path
import logging

from torch.utils.data import DataLoader

from datasets.classification import ClassificationCustomDataset
from datasets.segmentation import SegmentationCustomDataset
from datasets.classification.transforms import create_classification_transform
from datasets.segmentation.transforms import create_segmentation_transform
from datasets.utils.loader import create_loader
from datasets.classification.transforms import transforms_config


_logger = logging.getLogger(__name__)
_RECOMMEND_DATASET_DIR = "./datasets"


def build_dataset(args):

    _logger.info('-'*40)
    _logger.info('==> Loading data...')

    task = args.train.task
    data_dir = args.train.data

    assert Path(data_dir).exists(), \
        f"No such directory {data_dir}! It would be recommended as {_RECOMMEND_DATASET_DIR}"

    transform_func_for = {
        'classification': create_classification_transform,
        'segmentation': create_segmentation_transform
    }

    dataset_for = {
        'classification': ClassificationCustomDataset,
        'segmentation': SegmentationCustomDataset,
    }

    assert task in transform_func_for, f"The given task `{task}` is not supported!"
    assert task in dataset_for, f"The given task `{task}` is not supported!"

    train_transform = transform_func_for[task](is_training=True)
    eval_transform = transform_func_for[task](is_training=False)

    train_dataset = dataset_for[task](
        args, root=data_dir, split='train',
        transform=train_transform, target_transform=None  # TODO: apply target_transform
    )
    eval_dataset = dataset_for[task](
        args, root=data_dir, split='val',
        transform=eval_transform, target_transform=None  # TODO: apply target_transform
    )

    return train_dataset, eval_dataset


def build_dataloader(args, model, train_dataset, eval_dataset, profile):

    task = str(args.train.task).lower()
    if task == 'classification':
        collate_fn = None
        use_prefetcher = True

        train_data_cfg = transforms_config(is_train=True)
        setattr(model, "train_data_cfg", train_data_cfg)
        train_loader = create_loader(
            train_dataset,
            args.train.data,
            _logger,
            input_size=args.train.img_size,
            batch_size=args.train.batch_size,
            is_training=True,
            use_prefetcher=use_prefetcher,
            num_workers=args.environment.num_workers if not profile else 1,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=False,
            kwargs=train_data_cfg,
            args=args
        )

        val_data_cfg = transforms_config(is_train=False)
        setattr(model, "val_data_cfg", val_data_cfg)
        eval_loader = create_loader(
            eval_dataset,
            args.train.data,
            _logger,
            input_size=args.train.img_size,
            batch_size=args.train.batch_size,
            is_training=False,
            use_prefetcher=use_prefetcher,
            num_workers=args.environment.num_workers if not profile else 1,
            distributed=args.distributed,
            collate_fn=None,
            pin_memory=False,
            kwargs=val_data_cfg,
            args=args
        )
    elif task == 'segmentation':
        train_loader = DataLoader(train_dataset, batch_size=args.train.batch_size,
                                  num_workers=args.environment.num_workers if not profile else 1,
                                  collate_fn=None,
                                  pin_memory=False)
        eval_loader = DataLoader(eval_dataset, batch_size=args.train.batch_size,
                                 num_workers=args.environment.num_workers if not profile else 1,
                                 collate_fn=None,
                                 pin_memory=False)
    else:
        raise AssertionError(f"Task ({task}) is not understood!")
    return train_loader, eval_loader
