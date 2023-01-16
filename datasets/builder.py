from pathlib import Path
from functools import partial
import logging

import torch

from datasets.base import transforms_config
from datasets.utils.loader import fast_collate, PrefetchLoader
from datasets.classification import ClassificationCustomDataset
from datasets.segmentation import SegmentationCustomDataset
from datasets.classification.transforms import create_classification_transform
from datasets.utils.loader import init_worker


_logger = logging.getLogger(__name__)
_RECOMMEND_DATASET_DIR = "./datasets"


def build_dataset(args, rank=0, distributed=False):
    
    # TODO: DDP set-up at outer scope of this function
    if distributed and rank != 0:
        torch.distributed.barrier() # wait for rank 0 to download dataset
    
    _logger.info('-'*40)
    _logger.info('==> Loading data...')
        
    assert Path(args.data_dir).exists(), \
        f"No such directory {args.data_dir}! It would be recommended as {_RECOMMEND_DATASET_DIR}"

    train_dataset = ClassificationCustomDataset(
        root=args.data_dir, parser=args.dataset, split='train', args=args
        )
    eval_dataset = ClassificationCustomDataset(
        root=args.data_dir, parser=args.dataset, split='val', args=args
        )

    if distributed and rank == 0:
        torch.distributed.barrier()
    
    return train_dataset, eval_dataset

def create_loader(
        dataset,
        dataset_name,
        logger,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        re_mode='const',
        re_count=1,
        re_split=False,
        num_aug_repeats=0,
        num_aug_splits=0,
        num_workers=1,
        distributed=False,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        persistent_workers=True,
        worker_seeding='all',
        kwargs=None,
        args=None
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    kwargs['re_num_splits'] = re_num_splits

    dataset.transform = create_classification_transform(
        dataset_name,
        img_size=input_size[2],
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        **kwargs
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
    
    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else None

    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(init_worker, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            mean     = kwargs['mean'],
            std      = kwargs['std'],
            channels = input_size[1],
            fp16     = fp16,
            re_prob  = kwargs['re_prob']  if 're_prob' in kwargs else 0,
            re_mode  = kwargs['re_mode']  if 're_mode' in kwargs else 'const',
            re_count = kwargs['re_count'] if 're_count' in kwargs else 1,
            re_num_splits=re_num_splits
        )

    return loader

def build_dataloader(args, model, train_dataset, eval_dataset):

    collate_fn = None
    use_prefetcher = not args.no_prefetcher

    train_data_cfg = model.train_data_cfg if hasattr(model, 'train_data_cfg') else None
    train_data_cfg = transforms_config(args.dataset, True, train_data_cfg)
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

    val_data_cfg = model.val_data_cfg if hasattr(model, 'val_data_cfg') else None
    val_data_cfg = transforms_config(args.dataset, False, val_data_cfg)
    setattr(model, "val_data_cfg", val_data_cfg)
    val_loader = create_loader(
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
    return train_loader, val_loader

