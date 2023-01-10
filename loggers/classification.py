

MAX_LOGGING_SAMPLES = 5

class BaseLogger:
    def __init__(self) -> None:
        super(BaseLogger, self).__init__()
        self.epoch = 0
    
    def save_plot(self, image, pred):
        pass
    
    def append_csv(self, csv_logging, epoch):
        pass