"""
Based on the mobilevit official implementation.
https://github.com/apple/ml-cvnets/blob/6acab5e446357cc25842a90e0a109d5aeeda002f/cvnets/models/classification/mobilevit.py
"""

import argparse
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch import Tensor
# from . import register_cls_models
# from .base_cls import BaseEncoder
from models.configuration.mobilevit import get_configuration
from models.op.ml_cvnets import ConvLayer, LinearLayer, GlobalPool
from models.op.ml_cvnets import InvertedResidual, MobileViTBlock

__all__ = ['mobilevit']
SUPPORTING_TASK = ['classification']

# class MobileViT(BaseEncoder):
class MobileViT(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        # classifier_dropout = getattr(
        #     opts, "model.classification.classifier_dropout", 0.0
        # )

        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        image_channels = 3
        out_channels = 16

        mobilevit_config = get_configuration()

        super().__init__()
        
        """From BaseEncoder"""
        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.conv_1x1_exp = None
        self.classifier = None
        self.round_nearest = 8

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the backbone
        # We allow that using output_stride and replace_stride_with_dilation arguments
        self.dilation = 1
        output_stride = kwargs.get("output_stride", None)
        self.dilate_l4 = False
        self.dilate_l5 = False
        if output_stride == 8:
            self.dilate_l4 = True
            self.dilate_l5 = True
        elif output_stride == 16:
            self.dilate_l5 = True
        """End BaseEncoder"""

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer5"],
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=exp_channels,
            kernel_size=1,
            stride=1,
            use_act=True,
            use_norm=True,
        )

        self.model_conf_dict["exp_before_cls"] = {
            "in": in_channels,
            "out": exp_channels,
        }
        self.pool = GlobalPool(pool_type=pool_type, keep_dim=False)
        
        self._last_channels = exp_channels

        # self.classifier = nn.Sequential()
        # self.classifier.add_module(
        #     name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
        # )
        # if 0.0 < classifier_dropout < 1.0:
        #     self.classifier.add_module(
        #         name="dropout", module=nn.Dropout(p=classifier_dropout, inplace=True)
        #     )
        # self.classifier.add_module(
        #     name="fc",
        #     module=LinearLayer(
        #         in_features=exp_channels, out_features=num_classes, bias=True
        #     ),
        # )

        # # check model
        # self.check_model()

        # # weight initialization
        # self.reset_parameters(opts=opts)
        
    @property
    def last_channels(self):
        return self._last_channels

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.classification.mit.mode",
            type=str,
            default="small",
            choices=["xx_small", "x_small", "small"],
            help="MobileViT mode. Defaults to small",
        )
        group.add_argument(
            "--model.classification.mit.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mit.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between FFN layers. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mit.dropout",
            type=float,
            default=0.0,
            help="Dropout in Transformer layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mit.transformer-norm-layer",
            type=str,
            default="layer_norm",
            help="Normalization layer in transformer. Defaults to LayerNorm",
        )
        group.add_argument(
            "--model.classification.mit.no-fuse-local-global-features",
            action="store_true",
            help="Do not combine local and global features in MobileViT block",
        )
        group.add_argument(
            "--model.classification.mit.conv-kernel-size",
            type=int,
            default=3,
            help="Kernel size of Conv layers in MobileViT block",
        )

        group.add_argument(
            "--model.classification.mit.head-dim",
            type=int,
            default=None,
            help="Head dimension in transformer",
        )
        group.add_argument(
            "--model.classification.mit.number-heads",
            type=int,
            default=None,
            help="Number of heads in transformer",
        )
        return parser

    def _make_layer(
        self,
        opts,
        input_channel,
        cfg: Dict,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts, input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
        opts, input_channel: int, cfg: Dict, *args, **kwargs
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(
        self,
        opts,
        input_channel,
        cfg: Dict,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        # if transformer_dim % head_dim != 0:
        #     logger.error(
        #         "Transformer input dimension should be divisible by head dimension. "
        #         "Got {} and {}.".format(transformer_dim, head_dim)
        #     )

        block.append(
            MobileViTBlock(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.classification.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.classification.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(
                    opts, "model.classification.mit.attn_dropout", 0.1
                ),
                head_dim=head_dim,
                no_fusion=getattr(
                    opts,
                    "model.classification.mit.no_fuse_local_global_features",
                    False,
                ),
                conv_ksize=getattr(
                    opts, "model.classification.mit.conv_kernel_size", 3
                ),
            )
        )

        return nn.Sequential(*block), input_channel
    
    def _forward_layer(self, layer: nn.Module, x: Tensor) -> Tensor:
        # Larger models with large input image size may not be able to fit into memory.
        # We can use gradient checkpointing to enable training with large models and large inputs
        # return (
        #     gradient_checkpoint_fn(layer, x)
        #     if self.gradient_checkpointing
        #     else layer(x)
        # )
        return layer(x)

    def extract_end_points_all(
        self,
        x: Tensor,
        use_l5: Optional[bool] = True,
        use_l5_exp: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Dict[str, Tensor]:
        out_dict = {}  # Use dictionary over NamedTuple so that JIT is happy

        # if self.training and self.neural_augmentor is not None:
        #     x = self.neural_augmentor(x)
        #     out_dict["augmented_tensor"] = x

        x = self._forward_layer(self.conv_1, x)  # 112 x112
        x = self._forward_layer(self.layer_1, x)  # 112 x112
        out_dict["out_l1"] = x

        x = self._forward_layer(self.layer_2, x)  # 56 x 56
        out_dict["out_l2"] = x

        x = self._forward_layer(self.layer_3, x)  # 28 x 28
        out_dict["out_l3"] = x

        x = self._forward_layer(self.layer_4, x)  # 14 x 14
        out_dict["out_l4"] = x

        if use_l5:
            x = self._forward_layer(self.layer_5, x)  # 7 x 7
            out_dict["out_l5"] = x

            if use_l5_exp:
                x = self._forward_layer(self.conv_1x1_exp, x)
                out_dict["out_l5_exp"] = x
        return out_dict

    def extract_end_points_l4(self, x: Tensor, *args, **kwargs) -> Dict[str, Tensor]:
        return self.extract_end_points_all(x, use_l5=False)

    def _extract_features(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self._forward_layer(self.conv_1, x)
        x = self._forward_layer(self.layer_1, x)
        x = self._forward_layer(self.layer_2, x)
        x = self._forward_layer(self.layer_3, x)

        x = self._forward_layer(self.layer_4, x)
        x = self._forward_layer(self.layer_5, x)
        x = self._forward_layer(self.conv_1x1_exp, x)
        x = self._forward_layer(self.pool, x)
        return x

    def _forward_classifier(self, x: Tensor, *args, **kwargs) -> Dict:
        # We add another classifier function so that the classifiers
        # that do not adhere to the structure of BaseEncoder can still
        # use neural augmentor
        x = self._extract_features(x)
        # x = self.classifier(x)
        return x

    def forward(self, x: Any, *args, **kwargs) -> Dict:
        # if self.neural_augmentor is not None:
        #     if self.training:
        #         x_aug = self.neural_augmentor(x)
        #         prediction = self._forward_classifier(x_aug)  # .detach()
        #         out_dict = {"augmented_tensor": x_aug, "logits": prediction}
        #     else:
        #         out_dict = {
        #             "augmented_tensor": None,
        #             "logits": self._forward_classifier(x),
        #         }
        #     return out_dict
        # else:
        x = self._forward_classifier(x, *args, **kwargs)
        return {'last_feature': x}


def mobilevit(*args, **kwargs):
    return MobileViT(opts=None)