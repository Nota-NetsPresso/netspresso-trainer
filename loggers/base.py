

MAX_LOGGING_SAMPLES = 5

class BaseLogger:
    def __init__(self) -> None:
        super(BaseLogger, self).__init__()
        self._epoch = 0
    
    def log_epoch_end(self, image, pred, csv_logging, epoch=0):
        if epoch - self._epoch > 1 or epoch - self._epoch < 0:
            raise AssertionError(f"The given epoch ({epoch}) should be equal(or +1) to self._epoch {self._epoch}!")
        self._epoch += 1 if self._epoch != epoch else 0
        self.save_plot(image, pred)
        self.append_csv(csv_logging, epoch)
    
    def save_plot(self, image, pred):
        raise NotImplementedError
    
    def append_csv(self, csv_logging, epoch):
        raise NotImplementedError
