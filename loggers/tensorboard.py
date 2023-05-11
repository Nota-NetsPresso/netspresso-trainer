from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class TensorboardLogger:
    def __init__(self, task, model, result_dir, step_per_epoch: int, num_sample_images: int) -> None:
        super(TensorboardLogger, self).__init__()
        self.task = task
        self.model_name = model
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.step_per_epoch = step_per_epoch
        self.num_sample_images = num_sample_images
        
        self.tensorboard = SummaryWriter(self.result_dir / f"{self.task}_{self.model_name}")
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
        
    def log_scalar(self, key, value, mode='train'):
        global_step = self._epoch * self.step_per_epoch
        self.tensorboard.add_scalar(f"{mode}/{key}", value, global_step=global_step)
        
    def log_image(self, key, value: Union[np.ndarray, torch.Tensor], mode='train'):
        global_step = self._epoch * self.step_per_epoch
        value = self._as_numpy(value)
        self.tensorboard.add_image(f"{mode}/{key}", magic_image_handler(value), global_step=global_step, dataformats='HWC')


def magic_image_handler(img, num_example_image=1):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.transpose((1, 2, 0))
    elif img.ndim == 2:
        img = np.repeat(img[..., np.newaxis], 3, axis=2)
    elif img.ndim == 4:
        img = img[:min(img.shape[0], num_example_image)]  # first 4 batch
        img = np.concatenate(img, axis=-1)
        img = img.transpose((1, 2, 0))
    else:
        raise ValueError(f'img ndim is {img.ndim}, should be 2~4')

    min_, max_ = np.amin(img), np.amax(img)
    is_int_array = img.dtype in [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64]
    try_uint8 = (min_ >= 0 and max_ <= 255)

    if is_int_array and try_uint8:
        img = img.astype(np.uint8)
    else:
        if min_ >= 0 and max_ <= 1:
            img = (img * 255.0).astype(np.uint8)
        elif min_ >= -0.5 and max_ <= 0.5:
            img = ((img + 0.5) * 255.0).astype(np.uint8)
        elif min_ >= -1 and max_ <= 1:
            img = ((img + 1) / 2.0 * 255.0).astype(np.uint8)
        else:
            # denormalize with mean and std
            img = np.clip(img * (np.array(IMAGENET_DEFAULT_STD, dtype=np.float32) * 255.0) + np.array(IMAGENET_DEFAULT_MEAN, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)


    if img.shape[-1] != 1 and img.shape[-1] != 3:
        img = np.expand_dims(np.concatenate([img[..., i] for i in range(img.shape[-1])], axis=0), -1)
    img = np.clip(img, a_min=0, a_max=255)
    return img