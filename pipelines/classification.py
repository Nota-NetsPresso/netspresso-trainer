import os
from pathlib import Path
from typing import List
import time
from collections import deque


import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from tqdm import tqdm


from optimizers.builder import build_optimizer
from loggers.builder import build_logger
from pipelines.base import BasePipeline
from loggers.classification import ClassificationCSVLogger, ImageLogger
from utils.logger import set_logger

logger = set_logger('pipelines', level=os.getenv('LOG_LEVEL', default='INFO'))

MAX_SAMPLE_RESULT = 10
_RECOMMEND_CSV_LOG_PATH = "results.csv"
_RECOMMEND_OUTPUT_DIR = './'
_RECOMMEND_OUTPUT_DIR_NAME = 'exp'

class ClassificationPipeline(BasePipeline):
    def __init__(self, args, model, devices, train_dataloader, eval_dataloader, **kwargs):
        super(ClassificationPipeline, self).__init__(args, model, devices, train_dataloader, eval_dataloader, **kwargs)
        self.one_epoch_result = deque(maxlen=MAX_SAMPLE_RESULT)
    
    def set_train(self):
        
        assert self.model is not None
        self.optimizer = build_optimizer(self.model,
                                         opt=self.args.train.opt,
                                         lr=self.args.train.lr0,
                                         wd=self.args.train.weight_decay,
                                         momentum=self.args.train.momentum)
        
        output_dir = Path(_RECOMMEND_OUTPUT_DIR) / self.args.train.project / _RECOMMEND_OUTPUT_DIR_NAME
        output_dir.mkdir(exist_ok=True, parents=True)
        self.train_logger = build_logger(csv_path=output_dir / _RECOMMEND_CSV_LOG_PATH, task=self.args.train.task)
                
    def train_one_epoch(self):
        for idx, batch in enumerate(tqdm(self.train_dataloader, leave=False)):
            images, target = batch
            images = images.to(self.devices)
            target = target.to(self.devices)
            
            self.optimizer.zero_grad()
            with autocast():
                out = self.model(images)
                self.loss(out, target, mode='train')
                self.metric(out, target, mode='train')
            
            self.loss.backward()
            self.optimizer.step()
                        
            # # TODO: fn(out)
            # fn = lambda x: x
            # self.one_epoch_result.append(self.loss.result('train'))
            
            if self.args.distributed:
                torch.distributed.barrier()
        
        # self.one_epoch_result.clear()
        
    def validate(self):
        for idx, batch in enumerate(tqdm(self.eval_dataloader, leave=False)):
            images, target = batch
            images = images.to(self.devices)
            target = target.to(self.devices)
            
            with autocast():
                out = self.model(images)
                self.loss(out, target, mode='valid')
                self.metric(out, target, mode='valid')
            
            # self.one_epoch_result.append(self.loss.result('valid'))
            
            if self.args.distributed:
                torch.distributed.barrier()
        
        # self.one_epoch_result.clear()