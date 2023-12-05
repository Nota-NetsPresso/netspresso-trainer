from .csv import ClassificationCSVLogger, DetectionCSVLogger, SegmentationCSVLogger
from .visualizer import DetectionVisualizer, SegmentationVisualizer

CSV_LOGGER = {
    'classification': ClassificationCSVLogger,
    'segmentation': SegmentationCSVLogger,
    'detection': DetectionCSVLogger
}

VISUALIZER = {
    'segmentation': SegmentationVisualizer,
    'detection': DetectionVisualizer,
}
