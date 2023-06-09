from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Union
import math

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, task, model, result_dir, step_per_epoch: int, num_sample_images: int, args=None) -> None:
        super(TensorboardLogger, self).__init__()
        self.task = task
        self.model_name = model

        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.step_per_epoch = step_per_epoch
        self.num_sample_images = num_sample_images

        self.tensorboard = SummaryWriter(self.result_dir / "tensorboard")
        self._epoch = None

    def init_epoch(self):
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch = int(value)

    def _as_numpy(self, value: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            value = value.detach()
            value = value.cpu() if value.is_cuda else value
            value = value.numpy()
            return value

        raise TypeError(f"Unsupported type! {type(value)}")

    def _log_scalar(self, key: str, value, mode):
        global_step = self._epoch * self.step_per_epoch
        meta_string = f"{mode}/" if mode is not None else ""
        self.tensorboard.add_scalar(f"{meta_string}{key}", value, global_step=global_step)

    def _log_image(self, key: str, value: Union[np.ndarray, torch.Tensor], mode):
        global_step = self._epoch * self.step_per_epoch
        value = self._as_numpy(value)
        meta_string = f"{mode}/" if mode is not None else ""
        self.tensorboard.add_image(f"{meta_string}{key}", as_grid(value).astype(np.uint8),
                                   global_step=global_step, dataformats='HWC')

    def log_scalar(self, key, value, mode='train'):
        self._log_scalar(key, value, mode)

    def log_scalars_with_dict(self, scalar_dict, mode='train'):
        for k, v in scalar_dict.items():
            self._log_scalar(k, v, mode)

    def log_image(self, key, value: Union[np.ndarray, torch.Tensor], mode='train'):
        self._log_image(key, value, mode)

    def log_images_with_dict(self, image_dict, mode='train'):
        for k, v in image_dict.items():
            self._log_image(k, v, mode)

    def log_hparams(self, hp, final_metrics={}):
        # TODO: logging hp in recursive way
        hp_for_log = {key: value for key, value in dict(hp).items() if isinstance(value, float) or isinstance(value, int)}
        self.tensorboard.add_hparams(hp_for_log, metric_dict=final_metrics)

    def __call__(self,
                 train_losses, train_metrics, valid_losses, valid_metrics,
                 train_images, valid_images, learning_rate, elapsed_time,
                 ) -> None:

        self.log_scalars_with_dict(train_losses, mode='train')
        self.log_scalars_with_dict(train_metrics, mode='train')
        if train_images is not None:
            self.log_images_with_dict(train_images, mode='train')

        if valid_losses is not None:
            self.log_scalars_with_dict(valid_losses, mode='valid')
        if valid_metrics is not None:
            self.log_scalars_with_dict(valid_metrics, mode='valid')
        if isinstance(valid_images, dict):  # TODO: array with multiple dicts
            self.log_images_with_dict(valid_images, mode='valid')

        if learning_rate is not None:
            self.log_scalar('learning_rate', learning_rate, mode='misc')
        if elapsed_time is not None:
            self.log_scalar('elapsed_time', elapsed_time, mode='misc')


def _as_grid(image_batch, rows: int, cols: int):
    _, h_image, w_image, channels = image_batch.shape
    canvas = np.zeros((h_image * rows, w_image * cols, channels))
    for idx, image in enumerate(image_batch):
        idx_row = idx // cols
        idx_col = idx % cols
        canvas[idx_row*h_image:(idx_row+1)*h_image,
               idx_col*w_image:(idx_col+1)*w_image] = image
    return canvas


def as_grid(image_batch, rows: Optional[int] = None, cols: Optional[int] = None):
    num_images, _, _, _ = image_batch.shape
    if rows is not None and cols is not None:
        assert num_images <= rows * cols
        return _as_grid(image_batch, rows, cols)
    if rows is not None:
        cols = math.ceil(num_images / rows)
        return _as_grid(image_batch, rows, cols)
    if cols is not None:
        rows = math.ceil(num_images / cols)
        return _as_grid(image_batch, rows, cols)

    square_size = math.ceil(math.sqrt(num_images))
    rows, cols = square_size, square_size
    return _as_grid(image_batch, rows, cols)
