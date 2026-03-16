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

"""
Based on the YOLO implementation of WongKinYiu
https://github.com/WongKinYiu/YOLO/blob/main/yolo/model/module.py
"""
import math
import torch
import torch.nn as nn

from omegaconf import DictConfig
from torch import Tensor
from torch.fx.proxy import Proxy
from typing import Dict, List, Optional, Tuple, Union
from ....op.custom import Anchor2Vec, ConvLayer
from ....utils import ModelOutput
from netspresso_trainer.utils.bbox_utils import generate_anchors

def round_up(x: Union[int, Tensor], div: int = 1) -> Union[int, Tensor]:
    """
        Round up `x` to the biggest-nearest multiple of `div`
    """
    return x + (-x % div)


class Detection(nn.Module):
    """
        A single detection head.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_classes: int,
                 act_type: Optional[str] = None,
                 reg_max: Optional[int] = 16,
                 use_group: bool = True,
                 prior_prob: Optional[float] = 1e-2
        ):
        super().__init__()
        groups = 4 if use_group else 1
        reg_channels = 4 * reg_max
        reg_hidden_channels = max(round_up(hidden_channels // 4, groups), reg_channels, reg_max)
        cls_hidden_channels = max(hidden_channels, min(num_classes * 2, 128))

        self.reg_convs = nn.Sequential(
            ConvLayer(in_channels, reg_hidden_channels, kernel_size=3, act_type=act_type),
            ConvLayer(reg_hidden_channels, reg_hidden_channels, kernel_size=3, groups=groups, act_type=act_type),
            nn.Conv2d(reg_hidden_channels, reg_channels, kernel_size=1, groups=groups)
        )

        self.cls_convs = nn.Sequential(
            ConvLayer(in_channels, cls_hidden_channels, kernel_size=3, act_type=act_type),
            ConvLayer(cls_hidden_channels, cls_hidden_channels, kernel_size=3, act_type=act_type),
            nn.Conv2d(cls_hidden_channels, num_classes, kernel_size=1)
        )

        self.anchor2vec = Anchor2Vec(reg_max=reg_max)

        # Initialize
        def init_bn(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_bn)

        bias_reg = self.reg_convs[-1].bias.view(1, -1)
        bias_reg.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.reg_convs[-1].bias = torch.nn.Parameter(bias_reg.view(-1), requires_grad=True)

        bias_cls = self.cls_convs[-1].bias.view(1, -1)
        bias_cls.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.cls_convs[-1].bias = torch.nn.Parameter(bias_cls.view(-1), requires_grad=True)
    
    def forward(self, x: Union[Tensor, Proxy]) -> Tuple[Union[Tensor, Proxy]]:
        reg = self.reg_convs(x)
        cls_logits = self.cls_convs(x)
        output = torch.cat([reg, cls_logits], dim=1)
        return output


class YOLODetectionHead(nn.Module):
    def __init__(self,
                 num_classes: int,
                 intermediate_features_dim: List[int],
                 params: DictConfig):
        super().__init__()
        self._validate_params(params)
        self.num_classes = num_classes
        self.num_anchors = params.num_anchors
        self.hidden_dim = int(intermediate_features_dim[0])
        self.heads = self._build_heads(
            intermediate_features_dim,
            params.act_type,
            params.reg_max,
            params.use_group,
        )

        # if params.use_aux_loss:
        #     self.aux_heads = self._build_heads(
        #         intermediate_features_dim,
        #         params.act_type,
        #         params.reg_max,
        #         params.use_group,
        #     )
        # else:
        #     self.aux_heads = None

    def _validate_params(self, params: DictConfig) -> None:
        required_params = ['act_type', 'use_group', 'reg_max', 'num_anchors', 'use_aux_loss']
        for param in required_params:
            if not hasattr(params, param):
                raise ValueError(f"Missing required parameter: {param}")
    
    def _build_heads(
            self,
            intermediate_features_dim: List[int],
            act_type: str,
            reg_max: int,
            use_group: bool,
    ) -> nn.ModuleList:
        heads = nn.ModuleList()
        for feature_dim in intermediate_features_dim:
            head = Detection(
                int(feature_dim),
                self.hidden_dim,
                num_classes=self.num_classes,
                act_type=act_type,
                reg_max=reg_max,
                use_group=use_group,
            )
            heads.append(head)
        return heads

    def prepare_export(self, input_size: List[int],
                       feat_sizes: Optional[List[Tuple[int, int]]] = None) -> None:
        """Switch to export mode so forward() returns decoded (boxes, class_scores).

        Args:
            input_size: Model input image size as [height, width].
            feat_sizes: Optional list of (height, width) for each detection scale's
                feature map.  When provided the anchor grids are pre-computed and
                stored as constant buffers, which removes Range/Cast ops from the
                exported ONNX graph.
        """
        self._export = True
        self._export_input_size = input_size

        if feat_sizes is not None:
            h, w = input_size
            stage_strides = [w // fw for (_, fw) in feat_sizes]
            offset, scaler = generate_anchors((h, w), stage_strides)
            self.register_buffer('_export_offset', offset.float())
            self.register_buffer('_export_scaler', scaler.float())

    def forward(self, x_in: Union[List[Tensor], Dict]) -> ModelOutput:
        if isinstance(x_in, Dict):
            assert self.aux_heads
            aux_in = x_in["aux_outputs"]
            x_in = x_in["outputs"]
        else:
            aux_in = None
        outputs = [head(x) for head, x in zip(self.heads, x_in)]
        # if self.training and self.aux_heads:
        #     aux_outputs = [head(x) for head, x in zip(self.aux_heads, aux_in)]
        #     outputs = {"outputs": outputs, "aux_outputs": aux_outputs}

        if getattr(self, '_export', False):
            return self._decode_outputs(outputs)

        return ModelOutput(pred=outputs)

    def _decode_outputs(self, outputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Decode raw head outputs into merged (boxes_xyxy, class_scores) tensors.

        All detection scales are concatenated and processed by a *single* Anchor2Vec
        call (matching the expected ONNX structure), which keeps the graph compact.

        Args:
            outputs: List of per-scale tensors, each (B, 4*reg_max+num_classes, H, W).

        Returns:
            boxes: (B, total_anchors, 4) decoded xyxy box coordinates.
            class_scores: (B, total_anchors, num_classes) after sigmoid.
        """
        # Use the first head's Anchor2Vec; all heads share the same fixed weights.
        anchor2vec = self.heads[0].anchor2vec
        num_reg_channels = 4 * anchor2vec.reg_max

        pred_reg_flat: List[Tensor] = []
        pred_cls_flat: List[Tensor] = []

        for layer_output in outputs:
            reg = layer_output[:, :num_reg_channels]          # (B, 4*reg_max, H, W)
            cls = layer_output[:, num_reg_channels:]           # (B, num_classes, H, W)

            b, _, fh, fw = reg.shape
            pred_reg_flat.append(reg.reshape(b, num_reg_channels, fh * fw))

            b, c, fh, fw = cls.shape
            pred_cls_flat.append(cls.permute(0, 2, 3, 1).reshape(b, fh * fw, c))

        # (B, 4*reg_max, total_anchors) — one Softmax + Conv3d for all scales
        all_reg = torch.cat(pred_reg_flat, dim=2)
        # (B, total_anchors, num_classes) with sigmoid applied
        pred_class_logits = torch.cat(pred_cls_flat, dim=1).sigmoid()

        # Apply a single Anchor2Vec: unsqueeze trailing dim to make it 4-D (H=total_anchors, W=1)
        _, bbox_reg = anchor2vec(all_reg.unsqueeze(-1))  # (B, 4, total_anchors, 1)
        pred_bbox_reg = bbox_reg.squeeze(-1).permute(0, 2, 1)  # (B, total_anchors, 4)

        # Use pre-computed anchor buffers (no extra ONNX ops) when available,
        # otherwise fall back to on-the-fly computation.
        if hasattr(self, '_export_offset'):
            offset = self._export_offset
            scaler = self._export_scaler
        else:
            h, w = self._export_input_size
            stage_strides = [w // layer.shape[-1] for layer in outputs]
            offset, scaler = generate_anchors((h, w), stage_strides)
            device = outputs[0].device
            offset = offset.float().to(device)
            scaler = scaler.float().to(device)

        # Decode ltrb distances to xyxy coordinates
        pred_xyxy = pred_bbox_reg * scaler.view(1, -1, 1)
        lt, rb = pred_xyxy.chunk(2, dim=-1)
        boxes = torch.cat([offset.unsqueeze(0) - lt, offset.unsqueeze(0) + rb], dim=-1)

        return boxes, pred_class_logits

def yolo_detection_head(num_classes, intermediate_features_dim, conf_model_head, **kwargs):
    return YOLODetectionHead(num_classes=num_classes,
                             intermediate_features_dim=intermediate_features_dim,
                             params=conf_model_head.params)