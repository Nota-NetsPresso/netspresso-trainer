from abc import ABC, abstractmethod
import os
from statistics import mean

import torch

from losses.builder import build_losses
from metrics.builder import build_metrics
from utils.search_api import ModelSearchServerHandler
from utils.timer import Timer
from utils.logger import set_logger

logger = set_logger('pipelines', level=os.getenv('LOG_LEVEL', default='INFO'))
MAX_SAMPLE_RESULT = 10
START_EPOCH = 1

VALID_FREQ = 1
class BasePipeline(ABC):
    def __init__(self, args, model, devices, train_dataloader, eval_dataloader, is_online=True):
        super(BasePipeline, self).__init__()
        self.args = args
        self.model = model
        self.devices = devices
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        self.timer  = Timer()
        
        self.loss = None
        self.metric = None
        self.optimizer = None
        self.train_logger = None
        
        self.is_online = is_online
        if self.is_online:
            self.server_service = ModelSearchServerHandler(args.train.project, args.train.token)
            
    def _is_ready(self):
        assert self.model is not None, "`self.model` is not defined!"
        assert self.optimizer is not None, "`self.optimizer` is not defined!"
        assert self.train_logger is not None, "`self.train_logger` is not defined!"
        
    @abstractmethod
    def set_train(self):
        pass
    
    def train(self):
        logger.info("Train starts")
        logger.info(f"{self.args}")
        self.timer.start_record(name='train_all')
        self._is_ready()
        
        for num_epoch in range(START_EPOCH, self.args.train.epochs + START_EPOCH):
            self.timer.start_record(name=f'train_epoch_{num_epoch}')
            self.loss = build_losses(self.args)
            self.metric = build_metrics(self.args)
            self.train_one_epoch()  # append result in `self._one_epoch_result`
            
            self.timer.end_record(name=f'train_epoch_{num_epoch}')
            if num_epoch == START_EPOCH:  # FIXME: case for continuing training
                
                time_for_first_epoch = int(self.timer.get(name=f'train_epoch_{num_epoch}', as_pop=False))
                self.server_service.report_elapsed_time_for_epoch(time_for_first_epoch)
            logger.info(f"Epoch: {num_epoch} / {self.args.train.epochs}")
            logger.info(f"learning rate: {mean([param_group['lr'] for param_group in self.optimizer.param_groups]):.7f}")
            logger.info(f"training loss: {self.loss.result('train').get('total').avg:.7f}")
            logger.info(f"training metric: {[(name, value.avg) for name, value in self.metric.result('train').items()]}")
            
            if num_epoch % VALID_FREQ == START_EPOCH % VALID_FREQ:
                self.validate()
                logger.info(f"validation loss: {self.loss.result('valid').get('total').avg:.7f}")
                logger.info(f"validation metric: {[(name, value.avg) for name, value in self.metric.result('valid').items()]}")
            
            logger.info("-" * 40)

        self.timer.end_record(name='train_all')
        logger.info(f"Total time: {self.timer.get(name='train_all'):.2f} s")


    @abstractmethod
    def train_one_epoch(self):
        pass
    
    @torch.no_grad()
    @abstractmethod
    def validate(self):
        pass