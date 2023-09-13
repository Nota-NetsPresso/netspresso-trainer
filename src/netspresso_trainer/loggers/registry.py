from .csv import ClassificationCSVLogger, SegmentationCSVLogger
from .visualizer import DetectionVisualizer, SegmentationVisualizer

CSV_LOGGER = {
    'classification': ClassificationCSVLogger,
    'segmentation': SegmentationCSVLogger
}

VISUALIZER = {
    'segmentation': SegmentationVisualizer,
    'detection': DetectionVisualizer,
}