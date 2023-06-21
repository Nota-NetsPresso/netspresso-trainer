from pathlib import Path
import logging

from torch.utils.data import DataLoader

from datasets.classification import ClassificationCustomDataset
from datasets.segmentation import SegmentationCustomDataset
from datasets.detection import DetectionCustomDataset, detection_collate_fn
from datasets.classification.transforms import create_classification_transform
from datasets.segmentation.transforms import create_segmentation_transform
from datasets.detection.transforms import create_detection_transform
from datasets.utils.loader import create_loader


_logger = logging.getLogger(__name__)
_RECOMMEND_DATASET_DIR = "./datasets"


def build_dataset(args):

    _logger.info('-'*40)
    _logger.info('==> Loading data...')

    task = args.datasets.task
    data_dir = args.datasets.path.root

    assert Path(data_dir).exists(), \
        f"No such directory {data_dir}!"

    transform_func_for = {
        'classification': create_classification_transform,
        'segmentation': create_segmentation_transform,
        'detection': create_detection_transform
    }

    dataset_for = {
        'classification': ClassificationCustomDataset,
        'segmentation': SegmentationCustomDataset,
        'detection': DetectionCustomDataset,
    }

    assert task in transform_func_for, f"The given task `{task}` is not supported!"
    assert task in dataset_for, f"The given task `{task}` is not supported!"

    train_transform = transform_func_for[task](args, is_training=True)
    eval_transform = transform_func_for[task](args, is_training=False)

    train_dataset = dataset_for[task](
        args, root=data_dir, split='train',
        transform=train_transform, target_transform=None  # TODO: apply target_transform
    )
    eval_dataset = dataset_for[task](
        args, root=data_dir, split='val',
        transform=eval_transform, target_transform=None  # TODO: apply target_transform
    )

    _logger.info(f'Summary | Training dataset: {len(train_dataset)} sample(s)')
    _logger.info(f'Summary | Evaluation dataset: {len(eval_dataset)} sample(s)')

    return train_dataset, eval_dataset


def build_dataloader(args, task, model, train_dataset, eval_dataset, profile):

    if task == 'classification':
        collate_fn = None

        train_loader = create_loader(
            train_dataset,
            args.train.data,
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
            args.train.data,
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
            args.train.data,
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
            args.train.data,
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
            args.train.data,
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
            args.train.data,
            _logger,
            batch_size=args.training.batch_size if not args.distributed else 1,
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
