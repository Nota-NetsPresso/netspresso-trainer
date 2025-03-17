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
from typing import Dict, List, Literal, Optional

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
    def __init__(self, result_dir: str):
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

        mlflow.end_run()

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
        pass
