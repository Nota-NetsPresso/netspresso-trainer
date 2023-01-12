from collections import deque

from ..utils.search_api import ModelSearchServerHandler

MAX_SAMPLE_RESULT = 10

class BasePipeline:
    def __init__(self, args, is_online=True):
        super(BasePipeline, self).__init__()
        self.args = args
        
        self.dataloader = None
        self.model = None
        self.loss = None
        self.optimizer = None
        self.train_logger = None
        
        self.is_online = is_online
        if self.is_online:
            self.server_service = ModelSearchServerHandler(args.train.project_id, args.train.token)
    
    def train(self):
        assert self.dataloader is not None
        assert self.model is not None
        assert self.loss is not None
        assert self.optimizer is not None
        assert self.train_logger is not None
        
        one_epoch_result = deque(maxlen=MAX_SAMPLE_RESULT)
        for num_epoch in range(1, self.args.train.epochs + 1):
            one_epoch_result = self.train_one_epoch(one_epoch_result)  # append result in `self._one_epoch_result`
            one_epoch_result.clear()
            
            self.server_service.report_elapsed_time_for_epoch()
                    
    def train_one_epoch(self):
        raise NotImplementedError