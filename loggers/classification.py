from .base import BaseCSVLogger

CSV_HEADER = ['epoch', 'train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']

class ClassificationCSVLogger(BaseCSVLogger):
    def __init__(self, csv_path):
        super(ClassificationCSVLogger, self).__init__(csv_path)
        self.header = CSV_HEADER
        self.update_header()
        
class ImageLogger:
    def __init__(self) -> None:
        raise NotImplementedError