from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

from loggers.classification import ClassificationCSVLogger, ClassificationImageSaver
from loggers.segmentation import SegmentationCSVLogger, SegmentationImageSaver

CSV_LOGGER_TASK_SPECIFIC = {
    'classification': ClassificationCSVLogger,
    'segmentation': SegmentationCSVLogger
}

IMAGE_SAVER_TASK_SPECIFIC = {
    'classification': ClassificationImageSaver,
    'segmentation': SegmentationImageSaver
}

class TrainingLogger():
    def __init__(self, args, task: str, model: str, class_map: Dict, epoch: Optional[int]=None, step_per_epoch: Optional[int]=None) -> None:
        super(TrainingLogger, self).__init__()
        self.args = args
        self.task: str = task
        self.model: str = model
        self.class_map: Dict = class_map
        self.epoch: int = 1 if epoch is None else epoch
        self.step_per_epoch: Optional[int] = step_per_epoch
        
        self.use_tensorboard: bool = self.args.logging.tensorboard
        self.use_csvlogger: bool = self.args.logging.csv_logger
        self.use_imagesaver: bool = self.args.logging.image_saver
        
        self.csv_logger = CSV_LOGGER_TASK_SPECIFIC[task]()
        self.image_saver = CSV_LOGGER_TASK_SPECIFIC[task]()
    
    def log(self, train_losses, train_metrics, val_losses, val_metrics,
            train_images=None, val_images=None, learning_rate=None, elapsed_time=None):
        pass

# def build_logger(args, result_dir: Union[Path, str], csv_filename: Union[Path, str],
#                  task: str, model: str, class_map: Dict):
#     result_dir = Path(result_dir)
#     csv_path = result_dir / Path(csv_filename).with_suffix('.csv')
#     _task = task.lower()
    
#     pass
    
#     return inference_reporter

def build_logger(csv_path, task):
    if not task.lower() in CSV_LOGGER_TASK_SPECIFIC:
        raise AssertionError(f"No such task! (task: {task})")
    
    return CSV_LOGGER_TASK_SPECIFIC[task](csv_path)