from loggers.csv import ClassificationCSVLogger, SegmentationCSVLogger
from loggers.visualizer import SegmentationVisualizer, DetectionVisualizer


CSV_LOGGER = {
    'classification': ClassificationCSVLogger,
    'segmentation': SegmentationCSVLogger
}

VISUALIZER = {
    'segmentation': SegmentationVisualizer,
    'detection': DetectionVisualizer,
}