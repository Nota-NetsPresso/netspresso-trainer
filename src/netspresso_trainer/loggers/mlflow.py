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
import os
import numpy as np
import torch
from typing import Dict, List, Literal, Optional, Union

from loguru import logger

try:
    import mlflow
    if not hasattr(mlflow, "__version__"):
        raise ImportError("MLFlow is not installed. Please install it with `pip install mlflow`.")
except ImportError:
    raise ImportError(
        "MLFlow is not installed. Please install it with `pip install mlflow`."
        )


class MLFlowLogger:
    def __init__(self, result_dir: str, step_per_epoch: int):
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if uri is None:
            raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLFlow tracking URI: {uri}")

        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or "Default"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLFlow experiment name: {experiment_name}")

        run_name = os.environ.get("MLFLOW_RUN_NAME") or result_dir

        try:
            mlflow.start_run()
            logger.info(f"MLFlow run name: {run_name}")
        except Exception as e:
            logger.error(f"Failed to start MLFlow run: {e}")
        
        self.step_per_epoch = step_per_epoch


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

    def log_metrics_with_dict(self, scalar_dict, mode='train'):
        for k, v in scalar_dict.items():
            self._log_metric(k, v, mode)

    def _log_metric(self, key: str, value, mode):
        step = self._epoch * self.step_per_epoch
        meta_string = f"{mode}/" if mode is not None else ""
        mlflow.log_metric(f"{meta_string}{key}", value, step=step)

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
            self.log_metrics_with_dict(losses, mode=prefix)
        if metrics is not None:
            for k, v in metrics.items(): # Only mean values
                self._log_metric(k, v['mean'], mode=prefix)
        
        if learning_rate is not None:
            mlflow.log_metric("learning_rate", learning_rate) 
        if elapsed_time is not None:
            mlflow.log_metric("elapsed_time", elapsed_time)
        pass
