from .visualizer import ClassificationVisualizer, DetectionVisualizer, PoseEstimationVisualizer, SegmentationVisualizer

VISUALIZER = {
    'classification': ClassificationVisualizer,
    'segmentation': SegmentationVisualizer,
    'detection': DetectionVisualizer,
    'pose_estimation': PoseEstimationVisualizer,
}
