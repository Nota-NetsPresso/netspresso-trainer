import math
from typing import List, Tuple, Union

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....op.custom import ConvLayer
from ....utils import FXTensorListType, ModelOutput


class AllMLPDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            intermediate_features_dim: List[int],
            label_size: Union[Tuple[int, int], int],
            params: DictConfig,
        ):
        super().__init__()
        intermediate_channels = params.intermediate_channels
        classifier_dropout_prob = params.classifier_dropout_prob

        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for feature_dim in intermediate_features_dim:
            mlp = ConvLayer(feature_dim, intermediate_channels, kernel_size=1,
                            use_norm=True, use_act=True, norm_type='batch_norm', act_type='relu')
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = ConvLayer(len(intermediate_features_dim) * intermediate_channels,
                                     intermediate_channels, kernel_size=1,
                                     use_norm=True, use_act=False)

        self.dropout = nn.Dropout(classifier_dropout_prob)
        self.classifier = nn.Conv2d(intermediate_channels, num_classes, kernel_size=1)

        self.label_size = (label_size, label_size) if isinstance(label_size, int) else label_size

    def forward(self, encoder_hidden_states: FXTensorListType):

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):

            # unify channel dimension
            encoder_hidden_state = mlp(encoder_hidden_state)

            encoder_hidden_state = F.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )  # upsample to H/4 x W/4
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)  # B x {num_classes} x H/4 x W/4

        if self.label_size is not None:
            H, W = self.label_size[-2:]
            # upsample logits to the images' original size
            logits = F.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False
            )

        return ModelOutput(pred=logits)


def all_mlp_decoder(num_classes, intermediate_features_dim, label_size, conf_model_head, **kwargs) -> AllMLPDecoder:
    return AllMLPDecoder(num_classes,
                         intermediate_features_dim,
                         label_size=label_size,
                         params=conf_model_head.params)
