

import argparse
import os

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf

from optimizers.builder import build_optimizer
from schedulers import build_scheduler
from utils.logger import set_logger

logger = set_logger('train', level=os.getenv('LOG_LEVEL', 'INFO'))

IN_FEATURES = 10
OUT_FEATURES = 4


def parse_args_netspresso():

    parser = argparse.ArgumentParser(description="Parser for NetsPresso configuration")

    # -------- User arguments ----------------------------------------

    parser.add_argument(
        '--config', type=str, default='example-dev-segmentation.yaml',
        dest='config',
        help="Config path")

    parser.add_argument(
        '-o', '--output_dir', type=str, default='..',
        dest='output_dir',
        help="Checkpoint and result saving directory")

    parser.add_argument(
        '--profile', action='store_true',
        help="Whether to use profile mode")

    parser.add_argument(
        '--report-modelsearch-api', action='store_true',
        help="Report elapsed time for single epoch to NetsPresso Modelsearch API")

    args, _ = parser.parse_known_args()

    return args


class SampleDataset:
    def __init__(self, samples=100) -> None:
        self.samples = samples

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        return torch.FloatTensor(np.random.rand(IN_FEATURES)), torch.FloatTensor(np.random.rand(OUT_FEATURES))


if __name__ == '__main__':
    args_parsed = parse_args_netspresso()
    args = OmegaConf.load(args_parsed.config)

    model = nn.Linear(in_features=IN_FEATURES, out_features=OUT_FEATURES)
    model.cuda()

    optimizer = build_optimizer(model,
                                opt=args.train.opt,
                                lr=args.train.lr0,
                                wd=args.train.weight_decay,
                                momentum=args.train.momentum)

    dataloader = torch.utils.data.DataLoader(SampleDataset(samples=25), batch_size=5)

    sched_args = OmegaConf.create({
        'epochs': args.train.epochs,
        'lr_noise': None,
        'sched': 'poly',
        'decay_rate': args.train.schd_power,
        'min_lr': args.train.lrf, 
        'warmup_lr': 0.00001, # args.train.warmup_bias_lr
        'warmup_epochs': 5, # args.train.warmup_epochs
        'cooldown_epochs': 0,
    })

    scheduler, _ = build_scheduler(optimizer, sched_args)

    loss_func = nn.MSELoss()

    steps = 0
    for epoch in range(args.train.epochs):
        for x, y in dataloader:
            x = x.cuda()
            y = y.cuda()

            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out, y)
            loss.backward()

            optimizer.step()
            scheduler.step(epoch)
            steps += 1
        print(f"{steps} | {epoch}: ", [param['lr'] for param in optimizer.param_groups])
