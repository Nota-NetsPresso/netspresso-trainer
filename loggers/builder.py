from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

from loggers.base import BaseCSVLogger, BaseImageSaver
from loggers.classification import ClassificationCSVLogger, ClassificationImageSaver
from loggers.segmentation import SegmentationCSVLogger, SegmentationImageSaver
from loggers.tensorboard import TensorboardLogger
from loggers.stdout import StdOutLogger
from loggers.visualizer import VOCColorize

OUTPUT_ROOT_DIR = "./outputs"

CSV_LOGGER_TASK_SPECIFIC = {
    'classification': ClassificationCSVLogger,
    'segmentation': SegmentationCSVLogger
}

IMAGE_SAVER_TASK_SPECIFIC = {
    'classification': ClassificationImageSaver,
    'segmentation': SegmentationImageSaver
}

LABEL_CONVERTER_PER_TASK = {
    'segmentation': VOCColorize
}
class TrainingLogger():
    def __init__(self, args, task: str, model: str, class_map: Dict,
                 step_per_epoch: int, num_sample_images: int, epoch: Optional[int]=None) -> None:
        super(TrainingLogger, self).__init__()
        self.args = args
        self.task: str = task
        self.model: str = model
        self.class_map: Dict = class_map
        self.epoch = epoch
        
        result_dir: Path = Path(OUTPUT_ROOT_DIR) / self.args.train.project
        result_dir.mkdir(exist_ok=True)
        
        self.use_tensorboard: bool = self.args.logging.tensorboard
        self.use_csvlogger: bool = self.args.logging.csv
        self.use_imagesaver: bool = self.args.logging.image
        self.use_stdout: bool = self.args.logging.stdout
        
        self.csv_logger: Optional[BaseCSVLogger] = \
            CSV_LOGGER_TASK_SPECIFIC[task](model=model, result_dir=result_dir) if self.use_csvlogger else None
        self.image_saver: Optional[BaseImageSaver] = \
            IMAGE_SAVER_TASK_SPECIFIC[task](model=model, result_dir=result_dir) if self.use_imagesaver else None
        self.tensorboard_logger: Optional[TensorboardLogger] = \
            TensorboardLogger(task=task, model=model, result_dir=result_dir,
                              step_per_epoch=step_per_epoch, num_sample_images=num_sample_images) if self.use_tensorboard else None
        self.stdout_logger: Optional[StdOutLogger] = \
            StdOutLogger(task=task, model=model, total_epochs=args.train.epochs) if self.use_stdout else None
            
    def update_epoch(self, epoch: int):
        self.epoch = epoch
        if self.use_csvlogger:
            self.csv_logger.epoch = self.epoch
        if self.use_imagesaver:
            self.image_saver.epoch = self.epoch
        if self.use_tensorboard:
            self.tensorboard_logger.epoch = self.epoch
        if self.use_stdout:
            self.stdout_logger.epoch = self.epoch
            
    def _visualize_label(self, images):
        visualized_images = images # TODO: x 
        return visualized_images
    
    def log(self, train_losses, train_metrics, valid_losses=None, valid_metrics=None,
            train_images=None, valid_images=None, learning_rate=None, elapsed_time=None):
        if self.use_csvlogger:
            self.csv_logger(
                train_losses=train_losses,
                train_metrics=train_metrics,
                valid_losses=valid_losses,
                valid_metrics=valid_metrics
            )
        if self.use_imagesaver:
            self.image_saver(
                train_images=train_images,
                valid_images=valid_images
            )
        if self.use_tensorboard:
            self.tensorboard_logger(
                train_losses=train_losses,
                train_metrics=train_metrics,
                valid_losses=valid_losses,
                valid_metrics=valid_metrics,
                train_images=train_images,
                valid_images=valid_images,
                learning_rate=learning_rate,
                elapsed_time=elapsed_time
            )
        if self.use_stdout:
            self.stdout_logger(
                train_losses=train_losses,
                train_metrics=train_metrics,
                valid_losses=valid_losses,
                valid_metrics=valid_metrics,
                learning_rate=learning_rate,
                elapsed_time=elapsed_time
            )
            

def build_logger(args, task: str, model_name: str, step_per_epoch: int, class_map: Dict, num_sample_images: int, epoch: Optional[int]=None):
    training_logger = TrainingLogger(args,
                                     task=task.lower(), model=model_name.lower(),
                                     step_per_epoch=step_per_epoch,
                                     class_map=class_map, num_sample_images=num_sample_images,
                                     epoch=epoch)
    
    return training_logger