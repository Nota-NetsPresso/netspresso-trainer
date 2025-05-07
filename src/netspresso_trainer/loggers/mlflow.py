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
import glob
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf

try:
    import mlflow

    if not hasattr(mlflow, "__version__"):
        raise ImportError("MLFlow is not installed. Please install it with `pip install mlflow`.")
except Exception as e:
    logger.error(f"MLFlow is not installed. Please install it with `pip install mlflow`. Error: {e}")
    raise


class MLFlowLogger:
    def __init__(self, result_dir: str, step_per_epoch: int):
        self.step_per_epoch = step_per_epoch
        self.result_dir = result_dir
        self._setup_mlflow()

    def _setup_mlflow(self):
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if uri is None:
            raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLFlow tracking URI: {uri}")

        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or "Default"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLFlow experiment name: {experiment_name}")

        try:
            mlflow.start_run()
        except Exception as e:
            logger.error(f"Failed to start MLFlow run: {e}")

    def _as_numpy(self, value: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            value = value.detach()
            value = value.cpu() if value.is_cuda else value
            value = value.numpy()
            return value
        if isinstance(value, list):  # Pad images for tensorboard
            pad_shape = np.array([[v.shape[0], v.shape[1]] for v in value])
            pad_shape = pad_shape.max(0)
            ret_value = np.zeros((len(value), *pad_shape, 3), dtype=value[0].dtype)
            for i, v in enumerate(value):
                ret_value[i, : v.shape[0], : v.shape[1]] = v
            return ret_value

        raise TypeError(f"Unsupported type! {type(value)}")

    def log_metrics_with_dict(self, scalar_dict, mode="train"):
        for k, v in scalar_dict.items():
            self._log_metric(k, v, mode)

    def _log_metric(self, key: str, value, mode: Optional[str] = None):
        if "total" in key:
            return
        step = self._epoch * self.step_per_epoch
        meta_string = f"{mode}/" if mode is not None else ""
        mlflow.log_metric(f"{meta_string}{key}", value, step=step)

    def log_artifact(self):
        if not os.path.exists(self.result_dir):
            logger.warning(f"Artifact path {self.result_dir} does not exist.")
            return

        files = glob.glob(os.path.join(self.result_dir, "**"), recursive=True)
        for file in files:
            if os.path.isdir(file):
                continue

            skip_conditions = [
                "tensorboard" in file,
                "mlruns" in file,
                (file.endswith(".safetensors") and "best" not in file),
                (file.endswith(".pth") and "best" not in file),
            ]

            if any(skip_conditions):
                continue

            try:
                relative_path = os.path.relpath(file, self.result_dir)
                relative_dir = os.path.dirname(relative_path)
                mlflow.log_artifact(file, artifact_path=relative_dir)
            except Exception as e:
                logger.error(f"Failed to log artifact {file}: {e}")

    def log_onnx_model(self, model_path: Union[str, Path], input_example: Optional[torch.Tensor] = None) -> bool:
        """
        Log an ONNX model to MLflow.

        Args:
            model_path: Path to the ONNX model file
            input_example: Optional input tensor example for model signature

        Returns:
            bool: True if logging was successful, False otherwise
        """
        try:
            # Lazy import to avoid dependency issues
            import onnx

            # Load the ONNX model
            onnx_model = onnx.load_model(str(model_path))

            # Initialize signature as None
            signature = None

            # Process input example if provided
            if input_example is not None:
                if not isinstance(input_example, (torch.Tensor, np.ndarray)):
                    logger.warning(f"Input example must be a torch.Tensor or np.ndarray, got {type(input_example)}")
                else:
                    # Convert torch tensor to numpy if needed
                    if isinstance(input_example, torch.Tensor):
                        input_example = input_example.detach().cpu().numpy()

                    # Infer model signature from input example
                    try:
                        signature = mlflow.models.infer_signature(model_input=input_example)
                    except Exception as sig_error:
                        logger.warning(f"Failed to infer signature: {sig_error}")

            # Log the model to MLflow
            mlflow.onnx.log_model(onnx_model, "onnx_model", signature=signature)
            logger.info(f"Successfully logged ONNX model from {model_path}")
            return True

        except ImportError as ie:
            logger.error(f"Failed to import required modules for ONNX logging: {ie}")
            return False
        except Exception as e:
            logger.error(f"Failed to log ONNX model: {e}")
            return False

    def __call__(
        self,
        prefix: Literal["training", "validation", "evaluation", "inference"],
        epoch: Optional[int] = None,
        images: Optional[List] = None,
        losses: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        learning_rate: Optional[float] = None,
        elapsed_time: Optional[float] = None,
        **kwargs,
    ):
        self._epoch = 0 if epoch is None else epoch

        if losses is not None:
            self.log_metrics_with_dict(losses, mode=prefix)
        if metrics is not None:
            for k, v in metrics.items():  # Only mean values
                if "total" in k:
                    continue
                self._log_metric(k, v["mean"], mode=prefix)

        if learning_rate is not None:
            self._log_metric("learning_rate", learning_rate)
        if elapsed_time is not None:
            self._log_metric("elapsed_time", elapsed_time)
        if images is not None:
            pass

    def log_hparams(self, hp_omegaconf: OmegaConf):
        config_sections = {
            "training": self.flatten_dict,
            "environment": self.flatten_dict,
            "model": self.flatten_dict,
            "augmentation": self.flatten_augmentation_config,
        }

        for section_name, flatten_func in config_sections.items():
            # Check if the section exists in the config and access it safely
            if hasattr(hp_omegaconf, section_name):
                section_conf = getattr(hp_omegaconf, section_name)
                section_conf = OmegaConf.to_container(section_conf, resolve=True)
                flattened_conf = flatten_func(section_conf)
                mlflow.log_params(flattened_conf)

    def flatten_dict(self, nested_dict, parent_key="", sep="."):
        items = []
        for k, v in nested_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.extend(self._process_list_items(v, new_key, sep))
            else:
                items.append((new_key, v))

        return dict(items)

    def _process_list_items(self, list_items, parent_key, sep="."):
        result = []
        for i, item in enumerate(list_items):
            if isinstance(item, dict):
                result.extend(self.flatten_dict(item, f"{parent_key}{sep}{i}", sep=sep).items())
            else:
                result.append((f"{parent_key}{sep}{i}", item))
        return result

    def flatten_augmentation_config(self, aug_config):
        flat_dict = {}

        for phase in ["train", "inference"]:
            if phase not in aug_config:
                continue

            transforms = aug_config[phase]
            for transform in transforms:
                if "name" not in transform:
                    continue

                transform_name = transform["name"]
                for key, value in transform.items():
                    if key != "name":
                        flat_key = f"{phase}.{transform_name}.{key}"
                        flat_dict[flat_key] = value

        return flat_dict
