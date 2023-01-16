from pathlib import Path
import logging

from datasets.classification import ClassificationCustomDataset
from datasets.segmentation import SegmentationCustomDataset
from datasets.classification.transforms import create_classification_transform
from datasets.utils.loader import create_loader
from datasets.classification.transforms import transforms_config


_logger = logging.getLogger(__name__)
_RECOMMEND_DATASET_DIR = "./datasets"


def build_dataset(args):
    
    _logger.info('-'*40)
    _logger.info('==> Loading data...')
        
    assert Path(args.data_dir).exists(), \
        f"No such directory {args.data_dir}! It would be recommended as {_RECOMMEND_DATASET_DIR}"

    train_transform = create_classification_transform(
        args.dataset,
        img_size=args.input_size[2],
        is_training=True,
        use_prefetcher=not args.no_prefetcher
    )
    
    eval_transform = create_classification_transform(
        args.dataset,
        img_size=args.input_size[2],
        is_training=False,
        use_prefetcher=not args.no_prefetcher
    )

    train_dataset = ClassificationCustomDataset(
        root=args.data_dir, parser=args.dataset, split='train', args=args,
        transform=train_transform, target_transform=None  # TODO: apply target_transform
        )
    eval_dataset = ClassificationCustomDataset(
        root=args.data_dir, parser=args.dataset, split='val', args=args,
        transform=eval_transform, target_transform=None  # TODO: apply target_transform
        )
    
    return train_dataset, eval_dataset

def build_dataloader(args, model, train_dataset, eval_dataset):

    collate_fn = None
    use_prefetcher = not args.no_prefetcher

    train_data_cfg = transforms_config(is_train=True)
    setattr(model, "train_data_cfg", train_data_cfg)
    train_loader = create_loader(
        train_dataset,
        args.dataset,
        _logger,
        input_size=args.input,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=use_prefetcher,
        num_workers=args.environment.num_workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=False,
        kwargs = train_data_cfg,
        args = args
    )

    val_data_cfg = transforms_config(is_train=False)
    setattr(model, "val_data_cfg", val_data_cfg)
    eval_loader = create_loader(
        eval_dataset,
        args.dataset,
        _logger,
        input_size=args.input,
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=use_prefetcher,
        num_workers=args.environment.num_workers,
        distributed=args.distributed,
        collate_fn=None,
        pin_memory=False,
        kwargs = val_data_cfg,
        args = args
    )
    return train_loader, eval_loader

