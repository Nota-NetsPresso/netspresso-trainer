from loggers.base import BaseCSVLogger, BaseImageSaver

CSV_HEADER = ['epoch', 'train_loss', 'train_miou %', 'valid_miou %', 'valid_pixAcc %']


class SegmentationCSVLogger(BaseCSVLogger):
    def __init__(self, csv_path):
        super(SegmentationCSVLogger, self).__init__(csv_path)
        self.header = CSV_HEADER
        self.update_header()


class SegmentationImageSaver(BaseImageSaver):
    def __init__(self, result_dir) -> None:
        super(BaseImageSaver, self).__init__(result_dir)
    
    def save_result(self, data):
        return