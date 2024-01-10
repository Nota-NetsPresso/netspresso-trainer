from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image as Image
import torch
from omegaconf import OmegaConf

from ..utils.record import AverageMeter
from .base import BaseCSVLogger
from .image import ImageSaver
from .registry import CSV_LOGGER, VISUALIZER
from .stdout import StdOutLogger
from .tensorboard import TensorboardLogger
from .visualizer import magic_image_handler

START_EPOCH_ZERO_OR_ONE = 1

class TrainingLogger():
    def __init__(
        self,
        conf,
        task: Literal['classification', 'segmentation', 'detection'],
        model: str,
        class_map: Dict[int, str],
        step_per_epoch: int,
        num_sample_images: int,
        result_dir: Union[Path, str],
        epoch: Optional[int] = None,
    ) -> None:
        super(TrainingLogger, self).__init__()
        self.conf = conf
        self.task: str = task
        self.model: str = model
        self.class_map: Dict = class_map
        self.epoch = epoch
        self.num_sample_images = num_sample_images

        self.project_id = conf.logging.project_id if conf.logging.project_id is not None else f"{self.task}_{self.model}"

        self._result_dir = result_dir
        OmegaConf.save(config=self.conf, f=(result_dir / "hparams.yaml"))

        self.use_tensorboard: bool = self.conf.logging.tensorboard
        self.use_csvlogger: bool = self.conf.logging.csv
        self.use_imagesaver: bool = self.conf.logging.image
        self.use_stdout: bool = self.conf.logging.stdout
        self.use_netspresso: bool = False  # TODO: NetsPresso training board

        self.csv_logger: Optional[BaseCSVLogger] = \
            CSV_LOGGER[task](model=model, result_dir=self._result_dir) if self.use_csvlogger else None
        self.image_saver: Optional[ImageSaver] = \
            ImageSaver(model=model, result_dir=self._result_dir) if self.use_imagesaver else None
        self.tensorboard_logger: Optional[TensorboardLogger] = \
            TensorboardLogger(task=task, model=model, result_dir=self._result_dir,
                              step_per_epoch=step_per_epoch, num_sample_images=num_sample_images) if self.use_tensorboard else None
        self.stdout_logger: Optional[StdOutLogger] = \
            StdOutLogger(task=task, model=model, total_epochs=conf.training.epochs, result_dir=self._result_dir) if self.use_stdout else None

        self.netspresso_api_client = None
        if self.use_netspresso:
            from loggers.netspresso import ModelSearchServerHandler
            self.netspresso_api_client: Optional[ModelSearchServerHandler] = ModelSearchServerHandler(task=task, model=model)

        if task in VISUALIZER:
            pallete = conf.data.pallete if 'pallete' in conf.data else None
            self.label_converter = VISUALIZER[task](class_map=class_map, pallete=pallete)

    @property
    def result_dir(self):
        return self._result_dir

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
            if isinstance(v, (np.ndarray, float, int)):
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
        assert 'images' in images_dict
        image_new: np.ndarray = magic_image_handler(images_dict['images'])
        image_new = image_new.astype(np.uint8)
        images_dict.update({'images': image_new[:self.num_sample_images]})
        for k, v in images_dict.items():
            if k == 'images':
                continue

            # target, pred, bg_gt
            v = v[:self.num_sample_images]
            v_new: np.ndarray = magic_image_handler(
                self.label_converter(v, images=images_dict['images']))
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

    def log_end_of_traning(self, final_metrics=None):
        if final_metrics is None:
            final_metrics = {}
        if self.use_tensorboard:
            self.tensorboard_logger.log_hparams(self.conf, final_metrics=final_metrics)


def build_logger(conf, task: str, model_name: str, step_per_epoch: int, class_map: Dict[int, str], num_sample_images: int, result_dir: Union[Path, str], epoch: Optional[int] = None):
    training_logger = TrainingLogger(conf,
                                     task=task, model=model_name,
                                     step_per_epoch=step_per_epoch,
                                     class_map=class_map, num_sample_images=num_sample_images,
                                     result_dir=result_dir,
                                     epoch=epoch)

    return training_logger
