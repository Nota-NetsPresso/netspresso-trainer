import os
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from omegaconf import OmegaConf


from optimizers.builder import build_optimizer
from schedulers.builder import create_scheduler
from loggers.builder import build_logger
from pipelines.base import BasePipeline
from utils.logger import set_logger

logger = set_logger('pipelines', level=os.getenv('LOG_LEVEL', default='INFO'))

_RECOMMEND_CSV_LOG_PATH = "results.csv"
_RECOMMEND_OUTPUT_DIR = './'
_RECOMMEND_OUTPUT_DIR_NAME = 'exp'

CITYSCAPE_IGNORE_INDEX = 255  # TODO: get from configuration


class SegmentationPipeline(BasePipeline):
    def __init__(self, args, model, devices, train_dataloader, eval_dataloader, **kwargs):
        super(SegmentationPipeline, self).__init__(args, model, devices, train_dataloader, eval_dataloader, **kwargs)
        self.ignore_index = CITYSCAPE_IGNORE_INDEX
        self.num_classes = train_dataloader.dataset.num_classes

    def set_train(self):

        assert self.model is not None
        self.optimizer = build_optimizer(self.model,
                                         opt=self.args.train.opt,
                                         lr=self.args.train.lr0,
                                         wd=self.args.train.weight_decay,
                                         momentum=self.args.train.momentum)
        sched_args = OmegaConf.create({
            'epochs': self.args.train.epochs,
            'lr_noise': None,
            'sched': 'poly',
            'decay_rate': self.args.train.schd_power,
            'min_lr': 0,  # FIXME: add hyperparameter or approve to follow `self.args.train.lrf`
            'warmup_lr': 0.00001, # self.args.train.lr0
            'warmup_epochs': 5, # self.args.train.warmup_epochs
            'cooldown_epochs': 0,
        })
        
        self.scheduler, _ = create_scheduler(self.optimizer, sched_args)

        output_dir = Path(_RECOMMEND_OUTPUT_DIR) / self.args.train.project / _RECOMMEND_OUTPUT_DIR_NAME
        output_dir.mkdir(exist_ok=True, parents=True)
        self.train_logger = build_logger(csv_path=output_dir / _RECOMMEND_CSV_LOG_PATH, task=self.args.train.task)

    def train_step(self, batch):
        images, target = batch['pixel_values'], batch['labels']

        images = images.to(self.devices)
        target = target.long().to(self.devices)

        self.optimizer.zero_grad()
        out = self.model(images, label_size=target.size())
        self.loss(out, target, mode='train')

        self.loss.backward()
        self.optimizer.step()
        
        self.metric(out.detach(), target, mode='train')


        # # TODO: fn(out)
        # fn = lambda x: x
        # self.one_epoch_result.append(self.loss.result('train'))

        if self.args.distributed:
            torch.distributed.barrier()

    def valid_step(self, batch):
        images, target = batch['pixel_values'], batch['labels']
        images = images.to(self.devices)
        target = target.long().to(self.devices)

        out = self.model(images, label_size=target.size())
        self.loss(out, target, mode='valid')
        self.metric(out, target, mode='valid')

        if self.args.distributed:
            torch.distributed.barrier()

    def log_result(self, num_epoch, with_valid):
        logging_contents = {
            'epoch': num_epoch,
            'train_loss': self.train_loss,
            'train_miou %': self.metric.result('train').get('iou').avg,
        }

        if with_valid:
            logging_contents.update({
                'valid_miou %': self.metric.result('valid').get('iou').avg,
                'valid_pixAcc %': self.metric.result('valid').get('pixel_acc').avg
            })

        self.train_logger.update(logging_contents)
