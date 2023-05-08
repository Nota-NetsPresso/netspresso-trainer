from loggers.base import BaseCSVLogger, BaseImageSaver

CSV_HEADER = ['epoch', 'train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']

class ClassificationCSVLogger(BaseCSVLogger):
    def __init__(self, csv_path):
        super(ClassificationCSVLogger, self).__init__(csv_path)
        self.header = CSV_HEADER
        self.update_header()
        
class ClassificationImageSaver(BaseImageSaver):
    def __init__(self, result_dir) -> None:
        super(BaseImageSaver, self).__init__(result_dir)
    
    def save_result(self, data):
        return