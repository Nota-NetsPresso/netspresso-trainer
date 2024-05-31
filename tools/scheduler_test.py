# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from netspresso_trainer.optimizers import build_optimizer
from netspresso_trainer.schedulers import build_scheduler
from omegaconf import OmegaConf

IN_FEATURES = 10
OUT_FEATURES = 4


def parse_args():

    parser = argparse.ArgumentParser(description="Scheduler test configuration")

    parser.add_argument(
        '--training', type=str, default='config/training/template/common.yaml',
        dest='training',
        help="Config path")

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
    args_parsed = parse_args()
    conf = OmegaConf.load(args_parsed.training)
    conf_training = conf.training

    model = nn.Linear(in_features=IN_FEATURES, out_features=OUT_FEATURES)
    model.cuda()

    optimizer = build_optimizer(model,
                                opt=conf_training.opt,
                                lr=conf_training.lr,
                                wd=conf_training.weight_decay,
                                momentum=conf_training.momentum)

    dataloader = torch.utils.data.DataLoader(SampleDataset(samples=25), batch_size=5)
    scheduler, _ = build_scheduler(optimizer, conf_training)

    loss_func = nn.MSELoss()

    steps = 0
    for epoch in range(conf_training.epochs):
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
