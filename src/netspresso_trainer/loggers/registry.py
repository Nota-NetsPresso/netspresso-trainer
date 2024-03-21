from .visualizer import DetectionVisualizer, PoseEstimationVisualizer, SegmentationVisualizer

VISUALIZER = {
    'segmentation': SegmentationVisualizer,
    'detection': DetectionVisualizer,
    'pose_estimation': PoseEstimationVisualizer,
}
