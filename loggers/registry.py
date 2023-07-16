from loggers.classification import ClassificationCSVLogger
from loggers.segmentation import SegmentationCSVLogger
from loggers.visualizer import VOCColorize, DetectionVisualizer


CSV_LOGGER_TASK_SPECIFIC = {
    'classification': ClassificationCSVLogger,
    'segmentation': SegmentationCSVLogger
}

LABEL_CONVERTER_PER_TASK = {
    'segmentation': VOCColorize,
    'detection': DetectionVisualizer,
}