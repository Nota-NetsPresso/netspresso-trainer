from typing import List 
from omegaconf import DictConfig
import torch 
import torch.nn as nn 

from ....op.custom import ConvLayer, DWConvBlock
from ....utils import AnchorBasedDetectionModelOutput 
from .detection import AnchorGenerator


class YOLOFastestHeadV2(nn.Module): 
    def __init__(
        self,
        num_classes: int, 
        intermediate_features_dim: List[int], 
        params: DictConfig) -> None:
        super().__init__()


        self.reg_C2 = DWConvBlock(intermediate_features_dim[0], intermediate_features_dim[0], 5)
        self.cls_C2 = DWConvBlock(intermediate_features_dim[0], intermediate_features_dim[0], 5)

        self.reg_C3 = DWConvBlock(intermediate_features_dim[1], intermediate_features_dim[1], 5)
        self.cls_C3 = DWConvBlock(intermediate_features_dim[1], intermediate_features_dim[1], 5)
        self.num_layers = len(self.in_channels)
        self.cls_head = None  
        self.reg_head = None 
        self.obj_head = None


def yolo_fastest_head_v2(num_classes, intermediate_features_dim, conf_model_head) -> YOLOFastestHeadV2:
    return YOLOFastestHeadV2(num_classes=num_classes, 
                             intermediate_features_dim=intermediate_features_dim,
                             params=conf_model_head.params)


class YOLOFastestClassificationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class YOLOFastestRegressionHead(nn.Module): 
    def __init__(self) -> None:
        super().__init__()


class YOLOFastestObjectnessHead(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
