"""
Based on the RetinaNet implementation of torchvision.
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
"""
import math
from typing import List

from omegaconf import DictConfig
import torch
import torch.nn as nn

from ....op.custom import ConvLayer
from ....utils import AnchorBasedDetectionModelOutput
from .detection import AnchorGenerator


class AnchorDecoupledHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super().__init__()
        assert len(set(intermediate_features_dim)) == 1, "Feature dimensions of all stages have to same."
        in_channels = intermediate_features_dim[0]

        anchor_sizes = params.anchor_sizes
        aspect_ratios = params.aspect_ratios
        num_layers = params.num_layers

        norm_layer = params.norm_type

        aspect_ratios = (aspect_ratios,) * len(anchor_sizes)
        # TODO: Temporarily use hard-coded img_size
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios, (512, 512)) 
        num_anchors = self.anchor_generator.num_anchors_per_location()[0]

        self.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes, num_layers, norm_layer=norm_layer
        )
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors, num_layers, norm_layer=norm_layer)

    def forward(self, x):
        anchors = torch.cat(self.anchor_generator(x), dim=0)
        cls_logits = self.classification_head(x)
        bbox_regression = self.regression_head(x)
        return AnchorBasedDetectionModelOutput(anchors=anchors, cls_logits=cls_logits, bbox_regression=bbox_regression)


class RetinaNetClassificationHead(nn.Module):

    def __init__(
        self,
        in_channels,
        num_anchors,
        num_classes,
        num_layers,
        prior_probability = 0.01,
        norm_layer: str = 'batch_norm',
    ):
        super().__init__()

        conv = []
        for _ in range(num_layers):
            conv.append(
                ConvLayer(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=3, stride=1, norm_type=norm_layer)
            )
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return all_cls_logits


class RetinaNetRegressionHead(nn.Module):

    def __init__(
        self, 
        in_channels,
        num_anchors,
        num_layers,
        norm_layer: str = 'batch_norm'
    ):
        super().__init__()

        conv = []
        for _ in range(num_layers):
            conv.append(
                ConvLayer(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=3, stride=1, norm_type=norm_layer)
            )
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return all_bbox_regression


def anchor_decoupled_head(num_classes, intermediate_features_dim, conf_model_head, **kwargs) -> AnchorDecoupledHead:
    return AnchorDecoupledHead(num_classes=num_classes,
                               intermediate_features_dim=intermediate_features_dim,
                               params=conf_model_head.params)
