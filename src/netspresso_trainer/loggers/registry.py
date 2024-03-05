from .csv import ClassificationCSVLogger, DetectionCSVLogger, SegmentationCSVLogger
from .visualizer import DetectionVisualizer, SegmentationVisualizer, PoseEstimationVisualizer

CSV_LOGGER = {
    'classification': ClassificationCSVLogger,
    'segmentation': SegmentationCSVLogger,
    'detection': DetectionCSVLogger
}

VISUALIZER = {
    'segmentation': SegmentationVisualizer,
    'detection': DetectionVisualizer,
    'pose_estimation': PoseEstimationVisualizer,
}
