from typing import Optional, Tuple, List, Any, Union, Dict
import math
import itertools
import collections

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

from models.op.base_metaformer import MultiHeadAttention
from models.registry import NORM_REGISTRY, ACTIVATION_REGISTRY


class ViTPatchEmbeddings(nn.Module):
    """
    Refer to https://github.com/huggingface/transformers/blob/9a5d468ba0562e2d5edf9da787881fa227132bca/src/transformers/models/vit/modeling_vit.py#L146C1-L182C26
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, image_size, patch_size, in_channels, hidden_size):
        super().__init__()

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool) -> torch.Tensor:
        _, C, H, W = pixel_values.shape

        # (H, W) should be matched with self.image_size
        # assert H == self.image_size[0] and W == self.image_size[1]
        
        if C != self.in_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.in_channels} but got {C}."
            )
        if not interpolate_pos_encoding:
            if H != self.image_size[0] or W != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({H}*{W}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)  # B x (H*W) x C(=hidden_size)
        return embeddings

class ViTEmbeddings(nn.Module):
    """
    Refer to https://github.com/huggingface/transformers/blob/9a5d468ba0562e2d5edf9da787881fa227132bca/src/transformers/models/vit/modeling_vit.py#L66C1-L143C26
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, image_size, patch_size, in_channels, hidden_size, hidden_dropout_prob, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))  # C(=hidden_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if use_mask_token else None  # Optional[C(=hidden_size)]
        self.patch_embeddings = ViTPatchEmbeddings(image_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.patch_size = patch_size

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_size
        w0 = width // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape  # B x 3(={RGB}) x H x W
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)  # B x H'*W'(=num_patches) x C(=hidden_size)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]  # (H*W)
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)  # B x H'*W' x C
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask  # B x H'*W' x C

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # B x 1 x C
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # B x (H'*W' + 1) x C

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)  # B x (H'*W' + 1) x C
        else:
            embeddings = embeddings + self.position_embeddings  # B x (H'*W' + 1) x C

        embeddings = self.dropout(embeddings)  # B x (H'*W' + 1) x C

        return embeddings  # B x (H'*W' + 1) x C
    
    
class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, hidden_size, hidden_dropout_prob) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: B x S_s x C
        hidden_states = self.dense(hidden_states)  # B x S_s x C
        hidden_states = self.dropout(hidden_states)  # B x S_s x C

        return hidden_states
    
class ViTAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_scale, hidden_dropout_prob, output_with_attentions: bool = False) -> None:
        super().__init__()
        self.output_with_attentions = output_with_attentions
        self.attention = MultiHeadAttention(        
            hidden_size,
            num_attention_heads,
            attention_scale,
            output_with_attentions=self.output_with_attentions
        )
        self.output = ViTSelfOutput(hidden_size, hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # Let S: (H'*W' + 1)
        # hidden_states: B x S x C
        self_outputs = self.attention(hidden_states, head_mask=head_mask)  # B x S x C
        
        if self.output_with_attentions:
            attention_output = self_outputs[0]
        else:
            attention_output = self_outputs

        attention_output = self.output(attention_output, hidden_states)  # B x S_s x C_out(=C, =hidden_size)

        if self.output_with_attentions:
            outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
            return outputs

        outputs = attention_output
        return outputs  # B x S_s x C_out
    
    
class ViTIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACTIVATION_REGISTRY['gelu']()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class ViTOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_scale, layer_norm_eps, hidden_dropout_prob) -> None:
        super().__init__()
        self.seq_len_dim = 1
        self.attention = ViTAttention(hidden_size, num_attention_heads, attention_scale, hidden_dropout_prob)
        self.intermediate = ViTIntermediate(hidden_size, intermediate_size)
        self.output = ViTOutput(hidden_size, intermediate_size, hidden_dropout_prob)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # hidden_states: B x (H'*W' + 1) x C
        hidden_states = self.layernorm_before(hidden_states)  # B x (H'*W' + 1) x C
        self_attention_outputs = self.attention(
            hidden_states,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViTEncoder(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, attention_scale, layer_norm_eps, hidden_dropout_prob) -> None:
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(hidden_size, intermediate_size, num_attention_heads, attention_scale, layer_norm_eps, hidden_dropout_prob)
                                    for _ in range(num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, Dict]:
        # hidden_states: B x (H'*W' + 1) x C
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return dict(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class ViTPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class ViTModel(nn.Module):
    def __init__(
        self,
        num_hidden_layers,
        image_size,
        patch_size,
        in_channels,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        attention_scale,
        layer_norm_eps,
        hidden_dropout_prob,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False
    ):
        super().__init__()

        self.embeddings = ViTEmbeddings(image_size, patch_size, in_channels, hidden_size, hidden_dropout_prob, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, attention_scale, layer_norm_eps, hidden_dropout_prob)

        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.pooler = ViTPooler(hidden_size) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )  # B x (H'*W' + 1) x C

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return dict(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )