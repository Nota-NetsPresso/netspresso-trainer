"""
Based on the vit implementation of apple/ml-cvnets.
https://github.com/apple/ml-cvnets/blob/6acab5e446357cc25842a90e0a109d5aeeda002f/cvnets/models/classification/vit.py
"""
import argparse
from typing import Union, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor

from models.configuration.vit import get_configuration
from models.op.ml_cvnets import ConvLayer, LinearLayer, LayerNorm, PositionalEmbedding
from models.op.ml_cvnets import TransformerEncoder

__all__ = ['vit']
SUPPORTING_TASK = ['classification']
class VisionTransformer(nn.Module):
    """
    This class defines the `Vision Transformer architecture <https://arxiv.org/abs/2010.11929>`_. Our model implementation
    is inspired from `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    .. note::
        Our implementation is different from the original implementation in two ways:
        1. Kernel size is odd.
        2. Our positional encoding implementation allows us to use ViT with any multiple input scales
        3. We do not use StochasticDepth
        4. We do not add positional encoding to class token (if enabled), as suggested in `DeiT-3 paper <https://arxiv.org/abs/2204.07118>`_
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        image_channels = 3
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        pytorch_mha = getattr(opts, "model.classification.vit.use_pytorch_mha", False)

        vit_config = get_configuration()

        super().__init__()
        # if pytorch_mha and self.gradient_checkpointing:
        #     logger.error(
        #         "Current version of ViT supports PyTorch MHA without gradient checkpointing. "
        #         "Please use either of them, but not both"
        #     )
        
        """From BaseEncoder"""
        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.conv_1x1_exp = None
        # self.classifier = None
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
        
        kernel_sizes_conv_stem = [4, 2, 2]
        # Typically, in the ImageNet dataset, we use 224x224 as a resolution.
        # For out ViT implementation, patch size is 16 (16 = 4 * 2 * 2)
        # Therefore, total number of embeddings along width and height are (224 / 16)^2
        num_embeddings = (224 // 16) ** 2

        embed_dim = vit_config["embed_dim"]
        ffn_dim = vit_config["ffn_dim"]
        pos_emb_drop_p = vit_config["pos_emb_drop_p"]
        n_transformer_layers = vit_config["n_transformer_layers"]
        num_heads = vit_config["n_attn_heads"]
        attn_dropout = vit_config["attn_dropout"]
        dropout = vit_config["dropout"]
        ffn_dropout = vit_config["ffn_dropout"]
        norm_layer = vit_config["norm_layer"]

        conv_stem_proj_dim = max(32, embed_dim // 4)
        patch_emb = [
            ConvLayer(
                opts=opts,
                in_channels=image_channels,
                out_channels=conv_stem_proj_dim,
                kernel_size=kernel_sizes_conv_stem[0],
                stride=kernel_sizes_conv_stem[0],
                bias=False,
                use_norm=True,
                use_act=True,
            ),
            ConvLayer(
                opts=opts,
                in_channels=conv_stem_proj_dim,
                out_channels=conv_stem_proj_dim,
                kernel_size=kernel_sizes_conv_stem[1],
                stride=kernel_sizes_conv_stem[1],
                bias=False,
                use_norm=True,
                use_act=True,
            ),
            ConvLayer(
                opts=opts,
                in_channels=conv_stem_proj_dim,
                out_channels=embed_dim,
                kernel_size=kernel_sizes_conv_stem[2],
                stride=kernel_sizes_conv_stem[2],
                bias=True,
                use_norm=False,
                use_act=False,
            ),
        ]

        self.patch_emb = nn.Sequential(*patch_emb)

        use_cls_token = not getattr(
            opts, "model.classification.vit.no_cls_token", False
        )
        transformer_blocks = [
            TransformerEncoder(
                opts=opts,
                embed_dim=embed_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                transformer_norm_layer=norm_layer,
            )
            for _ in range(n_transformer_layers)
        ]

        # self.post_transformer_norm = get_normalization_layer(
        #     opts=opts, num_features=embed_dim, norm_type=norm_layer
        # )
        self.post_transformer_norm = LayerNorm(normalized_shape=embed_dim)

        self.transformer = nn.Sequential(*transformer_blocks)
        # self.classifier = LinearLayer(embed_dim, num_classes)

        # self.reset_parameters(opts=opts)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(size=(1, 1, embed_dim)))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.pos_embed = PositionalEmbedding(
            opts=opts,
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            sequence_first=False,
            padding_idx=None,
            is_learnable=not getattr(
                opts, "model.classification.vit.sinusoidal_pos_emb", False
            ),
            interpolation_mode="bilinear",
        )
        self.emb_dropout = nn.Dropout(p=pos_emb_drop_p)
        self.use_pytorch_mha = pytorch_mha
        self.embed_dim = embed_dim
        self.checkpoint_segments = getattr(
            opts, "model.classification.vit.checkpoint_segments", 4
        )

        self.model_conf_dict = {
            "conv1": {"in": image_channels, "out": embed_dim},
            "layer1": {"in": embed_dim, "out": embed_dim},
            "layer2": {"in": embed_dim, "out": embed_dim},
            "layer3": {"in": embed_dim, "out": embed_dim},
            "layer4": {"in": embed_dim, "out": embed_dim},
            "layer5": {"in": embed_dim, "out": embed_dim},
            "exp_before_cls": {"in": embed_dim, "out": embed_dim},
            "cls": {" in ": embed_dim, "out": num_classes},
        }

        # use_simple_fpn = getattr(opts, "model.classification.vit.use_simple_fpn", False)
        # self.simple_fpn = None
        # if use_simple_fpn:
        #     self.simple_fpn = self.build_simple_fpn_layers(opts, embed_dim)
        #     self.reset_simple_fpn_params()

        self.update_layer_norm_eps()
        
        self._last_channels = embed_dim


    def update_layer_norm_eps(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.eps = 1e-6

    # def reset_simple_fpn_params(self):
    #     for m in self.simple_fpn.modules():
    #         if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
    #             initialize_conv_layer(m, init_method="kaiming_uniform")

    # @staticmethod
    # def build_simple_fpn_layers(opts, embed_dim: int) -> nn.ModuleDict:
    #     layer_l2 = nn.Sequential(
    #         TransposeConvLayer(
    #             opts,
    #             in_channels=embed_dim,
    #             out_channels=embed_dim,
    #             kernel_size=2,
    #             stride=2,
    #             padding=0,
    #             output_padding=0,
    #             groups=1,
    #             use_norm=True,
    #             use_act=True,
    #         ),
    #         TransposeConvLayer(
    #             opts,
    #             in_channels=embed_dim,
    #             out_channels=embed_dim,
    #             kernel_size=2,
    #             stride=2,
    #             padding=0,
    #             output_padding=0,
    #             groups=1,
    #             use_norm=False,
    #             use_act=False,
    #             bias=True,
    #         ),
    #     )
    #     layer_l3 = TransposeConvLayer(
    #         opts,
    #         in_channels=embed_dim,
    #         out_channels=embed_dim,
    #         kernel_size=2,
    #         stride=2,
    #         padding=0,
    #         output_padding=0,
    #         groups=1,
    #         use_norm=False,
    #         use_act=False,
    #         bias=True,
    #     )
    #     layer_l4 = nn.Identity()
    #     layer_l5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    #     simple_fpn_layers = nn.ModuleDict(
    #         {
    #             "out_l2": layer_l2,
    #             "out_l3": layer_l3,
    #             "out_l4": layer_l4,
    #             "out_l5": layer_l5,
    #         }
    #     )

    #     return simple_fpn_layers
    
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
            "--model.classification.vit.mode",
            type=str,
            default="tiny",
            help="ViT mode. Default is Tiny",
        )
        group.add_argument(
            "--model.classification.vit.dropout",
            type=float,
            default=0.0,
            help="Dropout in ViT layers. Defaults to 0.0",
        )

        group.add_argument(
            "--model.classification.vit.norm-layer",
            type=str,
            default="layer_norm",
            help="Normalization layer in ViT",
        )

        group.add_argument(
            "--model.classification.vit.sinusoidal-pos-emb",
            action="store_true",
            help="Use sinusoidal positional encoding instead of learnable",
        )
        group.add_argument(
            "--model.classification.vit.no-cls-token",
            action="store_true",
            help="Do not use classification token",
        )
        group.add_argument(
            "--model.classification.vit.use-pytorch-mha",
            action="store_true",
            help="Use PyTorch's native multi-head attention",
        )

        group.add_argument(
            "--model.classification.vit.use-simple-fpn",
            action="store_true",
            help="Add simple FPN for down-stream tasks",
        )

        group.add_argument(
            "--model.classification.vit.checkpoint-segments",
            type=int,
            default=4,
            help="Number of checkpoint segments",
        )

        return parser

    def _extract_features(self, x: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            "ViT does not support feature extraction the same way as CNN."
        )

    def extract_patch_embeddings(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # input is of shape [Batch, in_channels, height, width]. in_channels is mostly 3 (for RGB images)
        batch_size = x.shape[0]

        # [Batch, in_channels, height, width] --> [Batch, emb_dim, num_patches_height, num_patches_width]
        patch_emb = self.patch_emb(x)
        n_h, n_w = patch_emb.shape[-2:]

        # [Batch, emb_dim, num_patches_height, num_patches_width] --> [Batch, emb_dim, num_patches]
        patch_emb = patch_emb.flatten(2)
        # [Batch, emb_dim, num_patches] --> [Batch, num_patches, emb_dim]
        patch_emb = patch_emb.transpose(1, 2).contiguous()

        n_patches = patch_emb.shape[1]
        pos_emb = self.pos_embed(n_patches).to(patch_emb.dtype)

        # add positional encodings
        patch_emb = pos_emb + patch_emb

        # add classification token
        if self.cls_token is not None:
            # [1, 1, emb_dim] --> [Batch, 1, emb_dim]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            # Concat([Batch, 1, emb_dim], [Batch, num_patches, emb_dim]) --> [Batch, num_patches + 1, emb_dim]
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

        # dropout
        patch_emb = self.emb_dropout(patch_emb)
        return patch_emb, (n_h, n_w)

    def _forward_classifier(self, x: Tensor, *args, **kwargs) -> Tensor:
        x, _ = self.extract_patch_embeddings(x)

        if self.use_pytorch_mha:
            # [B, N, C] --> [N, B, C]
            # For PyTorch MHA, we need sequence first.
            # For custom implementation, batch is the first
            x = x.transpose(0, 1)

        # if self.gradient_checkpointing:
        #     # we use sequential checkpoint function, which divides the model into chunks and checkpoints each segment
        #     # This maybe useful when dealing with large models

        #     # Right now, gradient checkpoint function does not like kwargs. Therefore, we use default MHA implementation
        #     # over the PyTorch's fused implementation.
        #     # Note that default MHA implementation is batch-first, while pytorch implementation is sequence-first.
        #     x = gradient_checkpoint_fn(self.transformer, self.checkpoint_segments, x)
        # else:
        for layer in self.transformer:
            x = layer(x, use_pytorch_mha=self.use_pytorch_mha)
        x = self.post_transformer_norm(x)

        # [N, B, C] or [B, N, C] --> [B, C]
        if self.cls_token is not None:
            x = x[0] if self.use_pytorch_mha else x[:, 0]
        else:
            x = torch.mean(x, dim=0) if self.use_pytorch_mha else torch.mean(x, dim=1)
        # [B, C] --> [B, Num_classes]
        # x = self.classifier(x)
        return x

    # def extract_end_points_all(
    #     self,
    #     x: Tensor,
    #     use_l5: Optional[bool] = True,
    #     use_l5_exp: Optional[bool] = False,
    #     *args,
    #     **kwargs
    # ) -> Dict[str, Tensor]:

    #     # if not self.simple_fpn:
    #     #     logger.error("Please enable simple FPN for down-stream tasks")

    #     # if self.cls_token:
    #     #     logger.error("Please disable cls token for down-stream tasks")

    #     # [Batch, 3, Height, Width] --> [Batch, num_patches + 1, emb_dim]
    #     batch_size, in_dim, in_height, in_width = x.shape

    #     out_dict = {}
    #     if self.training and self.neural_augmentor is not None:
    #         x = self.neural_augmentor(x)
    #         out_dict["augmented_tensor"] = x

    #     x, (n_h, n_w) = self.extract_patch_embeddings(x)

    #     # [Batch, num_patches, emb_dim] --> [Batch, num_patches, emb_dim]
    #     # if self.gradient_checkpointing:
    #     #     x = gradient_checkpoint_fn(
    #     #         self.transformer, self.checkpoint_segments, input=x
    #     #     )
    #     # else:
    #     x = self.transformer(x)

    #     x = self.post_transformer_norm(x)
    #     # [Batch, num_patches, emb_dim] --> [Batch, emb_dim, num_patches]
    #     x = x.transpose(1, 2)
    #     # [Batch, emb_dim, num_patches] --> [Batch, emb_dim, num_patches_h, num_patches_w]
    #     x = x.reshape(batch_size, x.shape[1], n_h, n_w)

    #     # build simple FPN, as suggested in https://arxiv.org/abs/2203.16527
    #     for k, extra_layer in self.simple_fpn.items():
    #         out_dict[k] = extra_layer(x)
    #     return out_dict

    # def profile_model(self, input: Tensor, *args, **kwargs) -> None:
    #     logger.log("Model statistics for an input of size {}".format(input.size()))
    #     logger.double_dash_line(dashes=65)
    #     print("{:>35} Summary".format(self.__class__.__name__))
    #     logger.double_dash_line(dashes=65)

    #     overall_params, overall_macs = 0.0, 0.0
    #     patch_emb, overall_params, overall_macs = self._profile_layers(
    #         self.patch_emb,
    #         input=input,
    #         overall_params=overall_params,
    #         overall_macs=overall_macs,
    #     )
    #     patch_emb = patch_emb.flatten(2)

    #     # [B, C, N] --> [B, N, C]
    #     patch_emb = patch_emb.transpose(1, 2)

    #     if self.cls_token is not None:
    #         # add classification token
    #         cls_tokens = self.cls_token.expand(patch_emb.shape[0], -1, -1)
    #         patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

    #     patch_emb, overall_params, overall_macs = self._profile_layers(
    #         self.transformer,
    #         input=patch_emb,
    #         overall_params=overall_params,
    #         overall_macs=overall_macs,
    #     )

    #     patch_emb, overall_params, overall_macs = self._profile_layers(
    #         self.classifier,
    #         input=patch_emb[:, 0],
    #         overall_params=overall_params,
    #         overall_macs=overall_macs,
    #     )

    #     logger.double_dash_line(dashes=65)
    #     print("{:<20} = {:>8.3f} M".format("Overall parameters", overall_params / 1e6))
    #     # Counting Addition and Multiplication as 1 operation
    #     print("{:<20} = {:>8.3f} M".format("Overall MACs", overall_macs / 1e6))
    #     overall_params_py = sum([p.numel() for p in self.parameters()])
    #     print(
    #         "{:<20} = {:>8.3f} M".format(
    #             "Overall parameters (sanity check)", overall_params_py / 1e6
    #         )
    #     )
    #     logger.double_dash_line(dashes=65)

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

def vit(*args, **kwargs):
    return VisionTransformer(opts=None)