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

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image as Image
import torch
from omegaconf import OmegaConf

from ..utils.record import AverageMeter
from .image import ImageSaver
from .registry import VISUALIZER
from .stdout import StdOutLogger
from .tensorboard import TensorboardLogger
from .visualizer import magic_image_handler


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
        self.use_imagesaver: bool = self.conf.logging.image
        self.use_stdout: bool = self.conf.logging.stdout

        self.loggers = []
        if self.use_imagesaver:
            self.loggers.append(ImageSaver(model=model, result_dir=self._result_dir))
        if self.use_tensorboard:
            self.tensorboard_logger = TensorboardLogger(task=task, model=model, result_dir=self._result_dir,
                                                        step_per_epoch=step_per_epoch, num_sample_images=num_sample_images)
            self.loggers.append(self.tensorboard_logger)
        if self.use_stdout:
            total_epochs = conf.training.epochs if hasattr(conf, 'training') else None
            self.loggers.append(StdOutLogger(task=task, model=model, total_epochs=total_epochs, result_dir=self._result_dir))

        if task in VISUALIZER:
            pallete = conf.data.pallete if 'pallete' in conf.data else None
            self.label_converter = VISUALIZER[task](class_map=class_map, pallete=pallete)

    @property
    def result_dir(self):
        return self._result_dir

    def update_epoch(self, epoch: int):
        self.epoch = epoch
        for logger in self.loggers:
            logger.epoch = self.epoch

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

            images_dict = {key: [] for key in images_list[0]}
            for minibatch in images_list:
                minibatch: Dict = self._convert_imagedict_as_readable(minibatch)
                for k_batch, v_batch in minibatch.items():
                    v_batch = list(v_batch)
                    if k_batch in images_dict:
                        images_dict[k_batch].extend(v_batch)
                        continue
                    images_dict[k_batch] = v_batch

            return images_dict

        if isinstance(images_dict_or_list, dict):
            images_dict = images_dict_or_list
            images_dict = self._convert_imagedict_as_readable(images_dict)
            return images_dict

        raise TypeError(f"Unsupported type for image logger!!! Current type: {type(images_dict_or_list)}")

    def log(
        self,
        prefix: Literal['training', 'validation', 'evaluation', 'inference'],
        epoch: Optional[int] = None,
        samples: Optional[List] = None,
        losses : Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        learning_rate: Optional[float] = None,
        elapsed_time: Optional[float] = None,
    ):
        if not self.use_imagesaver: # TODO: This is uneffective way
            samples = None

        if losses is not None:
            losses = self._convert_scalar_as_readable(losses)
        if metrics is not None:
            metrics = self._convert_scalar_as_readable(metrics)
        if samples is not None:
            samples = self._convert_images_as_readable(samples)

        for logger in self.loggers:
            logger(
                prefix=prefix,
                epoch=epoch,
                losses=losses,
                metrics=metrics,
                images=samples,
                learning_rate=learning_rate,
                elapsed_time=elapsed_time
            )

    def log_end_of_traning(self, final_metrics=None):
        if final_metrics is None:
            final_metrics = {}
        if self.use_tensorboard:
            self.tensorboard_logger.log_hparams(self.conf, final_metrics=final_metrics)
