from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import torch
import PIL.Image as Image

from loggers.base import BaseCSVLogger
from loggers.classification import ClassificationCSVLogger
from loggers.segmentation import SegmentationCSVLogger
from loggers.image import ImageSaver
from loggers.tensorboard import TensorboardLogger
from loggers.stdout import StdOutLogger
from loggers.netspresso import ModelSearchServerHandler
from loggers.visualizer import VOCColorize, magic_image_handler
from utils.common import AverageMeter

OUTPUT_ROOT_DIR = "./outputs"
START_EPOCH_ZERO_OR_ONE = 1


CSV_LOGGER_TASK_SPECIFIC = {
    'classification': ClassificationCSVLogger,
    'segmentation': SegmentationCSVLogger
}

LABEL_CONVERTER_PER_TASK = {
    'segmentation': VOCColorize
}


class TrainingLogger():
    def __init__(self, args, task: str, model: str, class_map: Dict,
                 step_per_epoch: int, num_sample_images: int, epoch: Optional[int] = None) -> None:
        super(TrainingLogger, self).__init__()
        self.args = args
        self.task: str = task
        self.model: str = model
        self.class_map: Dict = class_map
        self.epoch = epoch
        self.num_sample_images = num_sample_images

        self.project_id = args.train.project
        self.token = args.train.token

        result_dir: Path = Path(OUTPUT_ROOT_DIR) / self.project_id
        result_dir.mkdir(exist_ok=True, parents=True)

        self.use_tensorboard: bool = self.args.logging.tensorboard
        self.use_csvlogger: bool = self.args.logging.csv
        self.use_imagesaver: bool = self.args.logging.image
        self.use_stdout: bool = self.args.logging.stdout
        self.use_netspresso: bool = self.args.logging.netspresso

        self.csv_logger: Optional[BaseCSVLogger] = \
            CSV_LOGGER_TASK_SPECIFIC[task](model=model, result_dir=result_dir) if self.use_csvlogger else None
        self.image_saver: Optional[ImageSaver] = \
            ImageSaver(model=model, result_dir=result_dir) if self.use_imagesaver else None
        self.tensorboard_logger: Optional[TensorboardLogger] = \
            TensorboardLogger(task=task, model=model, result_dir=result_dir,
                              step_per_epoch=step_per_epoch, num_sample_images=num_sample_images) if self.use_tensorboard else None
        self.stdout_logger: Optional[StdOutLogger] = \
            StdOutLogger(task=task, model=model, total_epochs=args.training.epochs) if self.use_stdout else None
        self.netspresso_api_client: Optional[ModelSearchServerHandler] = \
            ModelSearchServerHandler(task=task, model=model) if self.use_netspresso else None
        if task in LABEL_CONVERTER_PER_TASK:
            self.label_converter = LABEL_CONVERTER_PER_TASK[task](class_map=class_map, pallete=args.data.pallete)

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
        if self.use_netspresso:
            self.netspresso_api_client.epoch = self.epoch

    @staticmethod
    def _to_numpy(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy()

    def _convert_scalar_as_readable(self, scalar_dict: Dict):
        for k, v in scalar_dict.items():
            if isinstance(v, np.ndarray) or isinstance(v, float) or isinstance(v, int):
                pass
                continue
            if isinstance(v, torch.Tensor):
                v_new = v.detach().cpu().numpy()
                scalar_dict.update({k: v_new})
                continue
            if isinstance(v, AverageMeter):
                v_new = v.avg
                scalar_dict.update({k: v_new})
                continue
            raise TypeError(f"Unsupported type for {k}!!! Current type: {type(v)}")
        return scalar_dict

    def _convert_imagedict_as_readable(self, images_dict: Dict):
        for k, v in images_dict.items():
            if len(v) > self.num_sample_images:
                v = v[:self.num_sample_images, ...]
            if 'images' in k:
                v_new: np.ndarray = magic_image_handler(v)
                v_new = v_new.astype(np.uint8)
                images_dict.update({k: v_new})
                continue
            # target, pred, bg_gt
            v_new: np.ndarray = magic_image_handler(self.label_converter(v))
            v_new = v_new.astype(np.uint8)
            images_dict.update({k: v_new})
        return images_dict

    def _convert_images_as_readable(self, images_dict_or_list: Union[Dict, List]):
        if isinstance(images_dict_or_list, list):
            images_list = images_dict_or_list

            if len(images_list) == 0:
                return None

            images_dict = {}
            for minibatch in images_list:
                minibatch: Dict = self._convert_imagedict_as_readable(minibatch)
                for k_batch, v_batch in minibatch.items():
                    if k_batch in images_dict:
                        images_dict[k_batch] = np.concatenate((images_dict[k_batch], v_batch), axis=0)
                        continue
                    images_dict[k_batch] = v_batch

            return images_dict

        if isinstance(images_dict_or_list, dict):
            images_dict = images_dict_or_list
            images_dict = self._convert_imagedict_as_readable(images_dict)
            return images_dict

        raise TypeError(f"Unsupported type for image logger!!! Current type: {type(images_dict_or_list)}")

    def log(self, train_losses, train_metrics, valid_losses=None, valid_metrics=None,
            train_images=None, valid_images=None, learning_rate=None, elapsed_time=None):
        train_losses = self._convert_scalar_as_readable(train_losses)
        train_metrics = self._convert_scalar_as_readable(train_metrics)

        if valid_losses is not None:
            valid_losses = self._convert_scalar_as_readable(valid_losses)
        if valid_metrics is not None:
            valid_metrics = self._convert_scalar_as_readable(valid_metrics)
        if train_images is not None:
            train_images = self._convert_images_as_readable(train_images)
        if valid_images is not None:
            valid_images = self._convert_images_as_readable(valid_images)

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
        if self.use_netspresso:
            # TODO: async handler if it takes much more time
            self.netspresso_api_client(
                train_losses=train_losses,
                train_metrics=train_metrics,
                valid_losses=valid_losses,
                valid_metrics=valid_metrics,
                learning_rate=learning_rate,
                elapsed_time=elapsed_time
            )

    def log_end_of_traning(self, final_metrics={}):
        if self.use_tensorboard:
            self.tensorboard_logger.log_hparams(self.args, final_metrics=final_metrics)


def build_logger(args, task: str, model_name: str, step_per_epoch: int, class_map: Dict, num_sample_images: int, epoch: Optional[int] = None):
    training_logger = TrainingLogger(args,
                                     task=task.lower(), model=model_name.lower(),
                                     step_per_epoch=step_per_epoch,
                                     class_map=class_map, num_sample_images=num_sample_images,
                                     epoch=epoch)

    return training_logger
