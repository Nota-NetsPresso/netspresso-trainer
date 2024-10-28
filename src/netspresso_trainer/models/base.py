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
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf

from .utils import BackboneOutput, DetectionModelOutput, ModelOutput


class TaskModel(nn.Module):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone: bool = False) -> None:
        super(TaskModel, self).__init__()
        self.task = conf_model.task
        self.name = conf_model.name
        self.backbone_name = conf_model.architecture.backbone.name
        if neck:
            self.neck_name = conf_model.architecture.neck.name
        self.head_name = conf_model.architecture.head.name

        self.backbone = backbone
        if neck:
            self.neck = neck
        self.head = head

        if freeze_backbone:
            self._freeze_backbone()
            logger.info(f"Freeze! {self.backbone_name} is now freezed. Now only tuning with {self.head_name}.")
        self.__save_dtype = None

    def _freeze_backbone(self):
        for m in self.backbone.parameters():
            m.requires_grad = False

    @property
    def head_list(self):
        return ('head')

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def save_dtype(self):
        return self.__save_dtype

    @save_dtype.setter
    def save_dtype(self, dtype):
        self.__save_dtype = dtype

    def _get_name(self):
        if hasattr(self, 'neck'):
            return f"{self.__class__.__name__}[task={self.task}, backbone={self.backbone_name}, neck={self.neck_name}, head={self.head_name}]"
        else:
            return f"{self.__class__.__name__}[task={self.task}, backbone={self.backbone_name}, head={self.head_name}]"

    @abstractmethod
    def forward(self, x, label_size=None, targets=None):
        raise NotImplementedError

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


class ClassificationModel(TaskModel):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone=False) -> None:
        super().__init__(conf_model, backbone, neck, head, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        out: ModelOutput = self.head(features['last_feature'])
        return out


class SegmentationModel(TaskModel):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone=False) -> None:
        super().__init__(conf_model, backbone, neck, head, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        if hasattr(self, 'neck'):
            features: BackboneOutput = self.neck(features['intermediate_features'])
        out: ModelOutput = self.head(features['intermediate_features'], targets)
        return out


class DetectionModel(TaskModel):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone=False) -> None:
        super().__init__(conf_model, backbone, neck, head, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        if hasattr(self, 'neck'):
            features: BackboneOutput = self.neck(features['intermediate_features'])
        out: DetectionModelOutput = self.head(features['intermediate_features'], targets)
        return out


class PoseEstimationModel(TaskModel):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone=False) -> None:
        super().__init__(conf_model, backbone, neck, head, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        if hasattr(self, 'neck'):
            features: BackboneOutput = self.neck(features['intermediate_features'])
        out: DetectionModelOutput = self.head(features['intermediate_features'], targets)
        return out


class ONNXModel:
    '''
        ONNX Model wrapper class for inferencing.
    '''
    def __init__(self, model_conf) -> None:
        import onnxruntime as ort
        self.name = model_conf.name + '_onnx'
        self.onnx_path = model_conf.checkpoint.path
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
            }),
            'CPUExecutionProvider',
        ]
        self.inference_session = ort.InferenceSession(model_conf.checkpoint.path, providers=providers)

    def _get_name(self):
        return f"{self.__class__.__name__}[model={self.name}]"

    def __call__(self, x, label_size=None, targets=None):
        device = x.device
        x = x.detach().cpu().numpy()
        out = self.inference_session.run(None, {self.inference_session.get_inputs()[0].name: x})
        out = [torch.tensor(o).to(device) for o in out]

        if len(out) == 1:
            out = out[0]

        return ModelOutput(pred=out)

    def eval(self):
        pass # Do nothing

    def set_provider(self, device):
        if device.type == 'cuda':
            self.inference_session.set_providers(['CUDAExecutionProvider'])
        else:
            self.inference_session.set_providers(['CPUExecutionProvider'])


class TFLiteModel:
    """
    TensorFlow Lite (tflite) wrapper class for inferencing.
    """

    NUM_THREADS = 4

    def __init__(self, model_conf) -> None:
        self.tflite = self._import_tflite()
        self.task = model_conf.task
        assert self.task == 'detection', f"Task {self.task} is not yet supported in this TensorFlow Lite (tflite) model inference."
        self.name = model_conf.name + '_tflite'
        self.tflite_path = model_conf.checkpoint.path
        self.interpreter = self.tflite.Interpreter(model_path=self.tflite_path, num_threads=self.NUM_THREADS)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype = self.input_details[0]['dtype']
        self.input_shape = tuple(self.input_details[0]['shape'])

        self.output_dtype = self.output_details[0]['dtype']
        self.quantized_input = self.input_dtype in [np.int8, np.uint8]
        self.quantized_output = self.output_dtype in [np.int8, np.uint8]
        if self.quantized_input:
            self.input_scale, self.input_zero_point = self.input_details[0]['quantization']

    def _import_tflite(self):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError as e:
                raise ImportError("Failed to import tensorflow lite. Please install tflite_runtime or tensorflow") from e
        return tflite

    def get_name(self) -> str:
        """Get the name of the model."""
        return f"{self.__class__.__name__}[model={self.name}]"

    def __call__(self, x: Union[np.ndarray, torch.Tensor], label_size=None, targets=None):
        """
        Perform inference on the input tensor.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input tensor
            label_size: Not used in this implementation
            targets: Not used in this implementation

        Returns:
            ModelOutput: Output of the model
        """
        try:
            device = x.device if hasattr(x, 'device') else 'cpu'
            x = self._prepare_input(x)

            if self.quantized_input:
                x = x / self.input_scale + self.input_zero_point
                x = x.astype(self.input_dtype)

            assert x.shape == self.input_shape, f"Your input shape {x.shape} does not match with the expected input shape {self.input_shape}"
            self.interpreter.set_tensor(self.input_details[0]['index'], x)
            self.interpreter.invoke()

            output = self._process_output(device)
            return ModelOutput(pred=output)
        except Exception as e:
            raise RuntimeError(f"Error during inference: {str(e)}") from e

    def _process_output(self, device: str) -> List[torch.Tensor]:
        """Process the output of the interpreter."""
        output = []
        for details in self.output_details:
            o = self.interpreter.get_tensor(details["index"])
            if self.quantized_output:
                output_quantization_params = details['quantization']
                o = (o.astype(np.float32) - output_quantization_params[1]) * output_quantization_params[0]
            output.append(torch.tensor(np.transpose(o, (0, 3, 1, 2))).to(device))

        if len(output) > 1:
            output.sort(key=lambda x: sum(x.shape), reverse=True)
        return output

    def eval(self):
        """Set the model to evaluation mode."""
        pass  # Do nothing for TFLite model

    def _prepare_input(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Prepare input tensor for inference.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input tensor

        Returns:
            np.ndarray: Prepared input tensor
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray):
            raise TypeError(f"Unsupported input type: {type(x)}")

        if x.shape[1] == 3:
            x = np.transpose(x, (0, 2, 3, 1))
        return x
