from loggers.base import BaseCSVLogger

CSV_HEADER = ['epoch', 'train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']

class ClassificationCSVLogger(BaseCSVLogger):
    def __init__(self, model, result_dir):
        super(ClassificationCSVLogger, self).__init__(model, result_dir)
        self.update_header(CSV_HEADER)
        
        self.key_map = {
            'epoch': 'epoch',
            'train/total': 'train_loss',
            'valid/total': 'valid_loss',
            'train/Acc@1': 'train_accuracy',
            'valid/Acc@1': 'valid_accuracy',
        }