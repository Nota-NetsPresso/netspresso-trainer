from loggers.classification import ClassificationCSVLogger
from loggers.segmentation import SegmentationCSVLogger


def build_logger(csv_path, task):
    if task.lower() == 'classification':
        return ClassificationCSVLogger(csv_path)
    elif task.lower() == 'segmentation':
        return SegmentationCSVLogger(csv_path)
    else:
        raise AssertionError(f"No such task! (task: {task})")


def build_visualizer():
    pass
