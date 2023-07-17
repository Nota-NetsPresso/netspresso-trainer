from typing import Optional, Tuple, List, Any, Union, Dict
import math
import itertools
import collections

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

    
class MetaFormerBlock(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.token_mixer = nn.Identity()  # TODO: define token mixer
        self.channel_mlp = nn.Identity()  # TODO: define channel nlp
    
    def forward(self, x):
        out_token_mixer = self.layernorm_before(x)
        out_token_mixer = self.token_mixer(out_token_mixer)
        
        out_token_mixer = out_token_mixer + x
        
        out_final = self.layernorm_after(out_token_mixer)
        out_final = self.channel_mlp(out_final)
        
        out_final = out_final + out_token_mixer
        
        return out_final
    
class MetaFormerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, layer_norm_eps) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MetaFormerBlock(hidden_size, layer_norm_eps) for _ in range(num_layers)]
        )
    
    def forward(self, x):
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
        return x

class MetaFormer(nn.Module):
    def __init__(self, num_layers, hidden_size, layer_norm_eps) -> None:
        super().__init__()
        self.patch_embed = nn.Identity()
        self.encoder = MetaFormerEncoder(num_layers, hidden_size, layer_norm_eps)
        self.norm = nn.Identity()
        
    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x
    
    def forward_tokens(self, x):
        x = self.encoder(x)
        return x
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x