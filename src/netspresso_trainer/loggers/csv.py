from .base import BaseCSVLogger


class ClassificationCSVLogger(BaseCSVLogger):
    csv_header = ['epoch', 'train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
    def __init__(self, model, result_dir):
        super(ClassificationCSVLogger, self).__init__(model, result_dir)
        self.update_header(self.csv_header)
        
        self._key_map = {
            'epoch': 'epoch',
            'train/total': 'train_loss',
            'valid/total': 'valid_loss',
            'train/Acc@1': 'train_accuracy',
            'valid/Acc@1': 'valid_accuracy',
        }
    
    @property
    def key_map(self):
        return self._key_map

class SegmentationCSVLogger(BaseCSVLogger):
    csv_header = ['epoch', 'train_loss', 'train_miou %', 'valid_miou %', 'valid_pixAcc %']
    def __init__(self, model, result_dir):
        super(SegmentationCSVLogger, self).__init__(model, result_dir)
        self.update_header(self.csv_header)
        
        self._key_map = {
            'epoch': 'epoch',
            'train/total': 'train_loss',
            'train/iou': 'train_miou %',
            'valid/iou': 'valid_miou %',
            'valid/pixel_acc': 'valid_pixAcc %',
        }

    @property
    def key_map(self):
        return self._key_map