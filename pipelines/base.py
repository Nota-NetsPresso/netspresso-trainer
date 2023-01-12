from collections import deque

from ..utils.search_api import ModelSearchServerHandler

MAX_SAMPLE_RESULT = 10

class BasePipeline:
    def __init__(self, args, model, devices, is_online=True):
        super(BasePipeline, self).__init__()
        self.args = args
        self.model = model
        self.devices = devices
        
        self.dataloader = None
        self.loss = None
        self.metric = None
        self.optimizer = None
        self.train_logger = None
        
        self.is_online = is_online
        if self.is_online:
            self.server_service = ModelSearchServerHandler(args.train.project_id, args.train.token)
            
    def _is_ready_to_train(self):
        assert self.dataloader is not None, "`self.dataloader` is not defined!"
        assert self.model is not None, "`self.model` is not defined!"
        assert self.loss is not None, "`self.loss` is not defined!"
        assert self.metric is not None, "`self.metric` is not defined!"
        assert self.optimizer is not None, "`self.optimizer` is not defined!"
        assert self.train_logger is not None, "`self.train_logger` is not defined!"
    
    def train(self):
        self._is_ready_to_train()
        
        one_epoch_result = deque(maxlen=MAX_SAMPLE_RESULT)
        for num_epoch in range(1, self.args.train.epochs + 1):
            one_epoch_result = self.train_one_epoch(one_epoch_result)  # append result in `self._one_epoch_result`
            one_epoch_result.clear()
            
            self.server_service.report_elapsed_time_for_epoch()
                    
    def train_one_epoch(self):
        raise NotImplementedError