from typing import List 
from omegaconf import DictConfig
import torch 
import torch.nn as nn 

from ....op.custom import ConvLayer
from ....utils import AnchorBasedDetectionModelOutput 
from .detection import AnchorGenerator


class YoloFastestHead(nn.Module):

    num_layers: int

    def __init__(
        self,
        num_classes: int,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super().__init__()

        anchors = params.anchors
        num_layers = len(anchors)
        self.anchors = anchors
        tmp_cell_anchors = []
        for a in self.anchors: 
            a = torch.tensor(a).view(-1, 2)
            wa = a[:, 0:1]
            ha = a[:, 1:]
            base_anchors = torch.cat([-wa, -ha, wa, ha], dim=-1)/2
            tmp_cell_anchors.append(base_anchors) 
        self.anchor_generator = AnchorGenerator(sizes=((128),)) # TODO: dynamic image_size, and anchor_size as a parameters
        self.anchor_generator.cell_anchors = tmp_cell_anchors
        num_anchors = self.anchor_generator.num_anchors_per_location()[0]
        self.num_anchors = num_anchors
        self.num_layers = num_layers
        self.num_classes = num_classes
        out_channels = num_anchors * (4 + num_classes) # TODO: Add confidence score dim 
        norm_type = params.norm_type
        use_act = False
        kernel_size = 1

        for i in range(num_layers):

            in_channels = intermediate_features_dim[i]

            conv_norm = ConvLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                use_act=use_act,
            )
            conv = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                use_norm=False,
                use_act=use_act,
            )

            layer = nn.Sequential(conv_norm, conv)

            setattr(self, f"layer_{i+1}", layer)

        def init_bn(M):
            for m in M.modules():

                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.apply(init_bn)

    def forward(self, inputs: List[torch.Tensor]):
        x1, x2 = inputs
        out1 = self.layer_1(x1)
        out2 = self.layer_2(x2)
        output = [out1, out2]
        all_cls_logits = []
        all_bbox_regression = []
        anchors = torch.cat(self.anchor_generator(output), dim=0)
        for idx, o in enumerate(output): 
            N, _, H, W = o.shape
            o = o.view(N, self.num_anchors, -1, H, W).permute(0, 3, 4, 1, 2)
            bbox_regression = o[..., :4]
            cls_logits = o[..., 4:]
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, K)
            all_bbox_regression.append(bbox_regression)
            all_cls_logits.append(cls_logits)
        return AnchorBasedDetectionModelOutput({"anchors": anchors, 
                                                "bbox_regression": all_bbox_regression, 
                                                "cls_logits": all_cls_logits, 
                                                })


def yolo_fastest_head(
    num_classes, intermediate_features_dim, conf_model_head, **kwargs
) -> YoloFastestHead:
    return YoloFastestHead(
        num_classes=num_classes,
        intermediate_features_dim=intermediate_features_dim,
        params=conf_model_head.params,
    )