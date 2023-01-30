import os
from pathlib import Path
from typing import List
import time
from collections import deque


import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from pipelines.base import BasePipeline
from utils.search_api import ModelSearchServerHandler
from loggers.classification import ClassificationCSVLogger, ImageLogger

MAX_SAMPLE_RESULT = 10
_RECOMMEND_CSV_LOG_PATH = "results.csv"
_RECOMMEND_OUTPUT_DIR = './'
_RECOMMEND_OUTPUT_DIR_NAME = 'exp'

class ClassificationPipeline(BasePipeline):
    def __init__(self, args, model, devices, train_dataloader, eval_dataloader, **kwargs):
        super(ClassificationPipeline, self).__init__(args, model, devices, train_dataloader, eval_dataloader, **kwargs)
        self.one_epoch_result = deque(maxlen=MAX_SAMPLE_RESULT)
    
    def set_train(self):
        self.loss = None
        self.metric = None
        self.optimizer = None
        
        output_dir = Path(_RECOMMEND_OUTPUT_DIR) / self.args.train.project / _RECOMMEND_OUTPUT_DIR_NAME
        output_dir.mkdir(exist_ok=True, parents=True)
        
        self.train_logger = ClassificationCSVLogger(csv_path=output_dir / _RECOMMEND_CSV_LOG_PATH)
                
    def train_one_epoch(self):
        
        for batch in self.train_dataloader:
            out = self.model(batch)
            # TODO: fn(out)
            fn = lambda x: x
            self.one_epoch_result.append(fn(out))
            
        self.one_epoch_result.clear()