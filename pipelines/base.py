from abc import ABC, abstractmethod

from losses.builder import build_losses
from metrics.builder import build_metrics
from utils.search_api import ModelSearchServerHandler
from utils.timer import Timer

MAX_SAMPLE_RESULT = 10

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
        self.timer.start_record(name='train_full')
        self._is_ready()
        
        for num_epoch in range(1, self.args.train.epochs + 1):
            self.timer.start_record(name=f'train_epoch_{num_epoch}')
            self.loss = build_losses(self.args)
            self.metric = build_metrics(self.args)
            self.train_one_epoch()  # append result in `self._one_epoch_result`
            
            self.timer.end_record(name=f'train_epoch_{num_epoch}')
            if num_epoch == 1:  # FIXME: case for continuing training
                time_for_first_epoch = int(self.timer.get(name=f'train_epoch_{num_epoch}', as_pop=False))
                self.server_service.report_elapsed_time_for_epoch(time_for_first_epoch)
            
        self.timer.end_record(name='train_full')

    @abstractmethod
    def train_one_epoch(self):
        pass