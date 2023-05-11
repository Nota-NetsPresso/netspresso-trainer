from loggers.base import BaseCSVLogger, BaseImageSaver

CSV_HEADER = ['epoch', 'train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']

class ClassificationCSVLogger(BaseCSVLogger):
    def __init__(self, model, result_dir):
        super(ClassificationCSVLogger, self).__init__(model, result_dir)
        self.update_header(CSV_HEADER)
                
class ClassificationImageSaver(BaseImageSaver):
    def __init__(self, model, result_dir) -> None:
        super(ClassificationImageSaver, self).__init__(model, result_dir)
    
    def save_result(self, data):
        return