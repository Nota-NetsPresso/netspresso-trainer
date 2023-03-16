from loggers.base import BaseCSVLogger

CSV_HEADER = ['epoch', 'train_loss', 'train_miou %', 'valid_miou %', 'valid_pixAcc %']


class SegmentationCSVLogger(BaseCSVLogger):
    def __init__(self, csv_path):
        super(SegmentationCSVLogger, self).__init__(csv_path)
        self.header = CSV_HEADER
        self.update_header()


class ImageLogger:
    def __init__(self) -> None:
        raise NotImplementedError
