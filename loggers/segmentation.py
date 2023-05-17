import numpy as np
import torch

from loggers.base import BaseCSVLogger, BaseImageSaver


CSV_HEADER = ['epoch', 'train_loss', 'train_miou %', 'valid_miou %', 'valid_pixAcc %']


class SegmentationCSVLogger(BaseCSVLogger):
    def __init__(self, model, result_dir):
        super(SegmentationCSVLogger, self).__init__(model, result_dir)
        self.update_header(CSV_HEADER)
        
        self.key_map = {
            'epoch': 'epoch',
            'train/total': 'train_loss',
            'train/iou': 'train_miou %',
            'valid/iou': 'valid_miou %',
            'valid/pixel_acc': 'valid_pixAcc %',
        }

class SegmentationImageSaver(BaseImageSaver):
    def __init__(self, model, result_dir) -> None:
        super(SegmentationImageSaver, self).__init__(model, result_dir)
    