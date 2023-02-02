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
        
    assert Path(args.train.data).exists(), \
        f"No such directory {args.train.data}! It would be recommended as {_RECOMMEND_DATASET_DIR}"

    train_transform = create_classification_transform(
        args.train.data,
        img_size=args.train.img_size,
        is_training=True,
        use_prefetcher=True
    )
    
    eval_transform = create_classification_transform(
        args.train.data,
        img_size=args.train.img_size,
        is_training=False,
        use_prefetcher=True
    )

    train_dataset = ClassificationCustomDataset(
        args=args, root=args.train.data, split='train', 
        parser=args.train.data, 
        transform=train_transform, target_transform=None  # TODO: apply target_transform
        )
    eval_dataset = ClassificationCustomDataset(
        args=args, root=args.train.data, split='val', 
        parser=args.train.data,
        transform=eval_transform, target_transform=None  # TODO: apply target_transform
        )
    
    return train_dataset, eval_dataset

def build_dataloader(args, model, train_dataset, eval_dataset, profile):

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
        kwargs = train_data_cfg,
        args = args
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
        kwargs = val_data_cfg,
        args = args
    )
    return train_loader, eval_loader

