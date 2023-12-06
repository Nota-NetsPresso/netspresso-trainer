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
    csv_header = ['epoch', 'train_loss', 'valid_loss', 'train_miou', 'valid_miou', 'train_pixel_accuracy', 'valid_pixel_accuracy']
    def __init__(self, model, result_dir):
        super(SegmentationCSVLogger, self).__init__(model, result_dir)
        self.update_header(self.csv_header)

        self._key_map = {
            'epoch': 'epoch',
            'train/total': 'train_loss',
            'valid/total': 'valid_loss',
            'train/iou': 'train_miou',
            'valid/iou': 'valid_miou',
            'train/pixel_acc': 'train_pixel_accuracy',
            'valid/pixel_acc': 'valid_pixel_accuracy',
        }

    @property
    def key_map(self):
        return self._key_map


class DetectionCSVLogger(BaseCSVLogger):
    csv_header = ['epoch', 'train_loss', 'valid_loss', 'train_map50', 'valid_map50', 'train_map75', 'valid_map75', 'train_map50_95', 'valid_map50_95']
    def __init__(self, model, result_dir):
        super(DetectionCSVLogger, self).__init__(model, result_dir)
        self.update_header(self.csv_header)

        self._key_map = {
            'epoch': 'epoch',
            'train/total': 'train_loss',
            'valid/total': 'valid_loss',
            'train/map50': 'train_map50',
            'valid/map50': 'valid_map50',
            'train/map75': 'train_map75',
            'valid/map75': 'valid_map75',
            'train/map50_95': 'train_map50_95',
            'valid/map50_95': 'valid_map50_95',
        }

    @property
    def key_map(self):
        return self._key_map
