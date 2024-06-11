# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class TensorboardLogger:
    def __init__(self, task, model, result_dir, step_per_epoch: int, num_sample_images: int) -> None:
        super(TensorboardLogger, self).__init__()
        self.task = task
        self.model_name = model

        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.step_per_epoch = step_per_epoch
        self.num_sample_images = num_sample_images

        self.tensorboard = SummaryWriter(self.result_dir / "tensorboard")

    def _as_numpy(self, value: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            value = value.detach()
            value = value.cpu() if value.is_cuda else value
            value = value.numpy()
            return value
        if isinstance(value, list): # Pad images for tensorboard
            pad_shape = np.array([[v.shape[0], v.shape[1]] for v in value])
            pad_shape = pad_shape.max(0)
            ret_value = np.zeros((len(value), *pad_shape, 3), dtype=value[0].dtype)
            for i, v in enumerate(value):
                ret_value[i, :v.shape[0], :v.shape[1]] = v
            return ret_value

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

    def _get_rasterized_hparam(self, hparams):
        if not isinstance(hparams, dict):
            stem = hparams
            if not isinstance(hparams, (int, float, str, bool, torch.Tensor)):
                return str(stem)
            return stem

        rasterized_dict = {}
        for key, value in hparams.items():
            if isinstance(value, dict):
                rasterized_value = self._get_rasterized_hparam({f"{key}/{k}": v for k, v in value.items()})
                for k, v in rasterized_value.items():
                    rasterized_dict[k] = v
                continue
            rasterized_value = self._get_rasterized_hparam(value)
            rasterized_dict[key] = rasterized_value
        return rasterized_dict

    def log_hparams(self, hp_omegaconf: Union[Dict, List], final_metrics=None):

        if final_metrics is None:
            final_metrics = {}
        final_metrics = {f"hparams_metrics/{k}": v for k, v in final_metrics.items()}

        hp_dict = OmegaConf.to_container(hp_omegaconf, resolve=True)
        hp_for_log = self._get_rasterized_hparam(hp_dict)

        exp, ssi, sei = hparams(hparam_dict=hp_for_log, metric_dict=final_metrics)
        self.tensorboard.file_writer.add_summary(exp)
        self.tensorboard.file_writer.add_summary(ssi)
        self.tensorboard.file_writer.add_summary(sei)
        for k, v in final_metrics.items():
            self.tensorboard.add_scalar(k, v)

    def __call__(
        self,
        prefix: Literal['training', 'validation', 'evaluation', 'inference'],
        epoch: Optional[int] = None,
        images: Optional[List] = None,
        losses : Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        learning_rate: Optional[float] = None,
        elapsed_time: Optional[float] = None,
        **kwargs
    ):
        self._epoch = 0 if epoch is None else epoch
        if losses is not None:
            self.log_scalars_with_dict(losses, mode=prefix)
        if metrics is not None:
            self.log_scalars_with_dict(metrics, mode=prefix)
        if isinstance(images, dict):  # TODO: array with multiple dicts
            self.log_images_with_dict(images, mode=prefix)

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
