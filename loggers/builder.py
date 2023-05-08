from pathlib import Path
from typing import Union

from loggers.base import BaseCSVLogger, BaseVisualizer, InferenceReporter
from loggers.classification import ClassificationCSVLogger, ClassificationVisualizer
from loggers.segmentation import SegmentationCSVLogger, SegmentationVisualizer

CSV_LOGGER_TASK_SPECIFIC = {
    'classificaiton': ClassificationCSVLogger,
    'segmentation': SegmentationCSVLogger
}

VISUALIZER_TASK_SPECIFIC = {
    'classificaiton': ClassificationVisualizer,
    'segmentation': SegmentationVisualizer
}

def build_logger(result_dir: Union[Path, str], csv_filename: Union[Path, str], task):
    result_dir = Path(result_dir)
    csv_path = result_dir / Path(csv_filename).with_suffix('.csv')
    _task = task.lower()
    
    csv_logger: BaseCSVLogger = CSV_LOGGER_TASK_SPECIFIC[_task](csv_path)
    visualizer: BaseVisualizer = VISUALIZER_TASK_SPECIFIC[_task](result_dir)
    inference_reporter = InferenceReporter(csv_logger=csv_logger, visualizer=visualizer)
    
    return inference_reporter