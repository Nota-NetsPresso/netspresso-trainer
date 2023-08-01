from abc import abstractmethod
from typing import List, TypedDict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

class BackboneOutput(TypedDict):
    intermediate_features: Optional[Union[List[Tensor], List[Proxy]]]
    last_feature: Optional[Union[Tensor, Proxy]]
    
class ModelOutput(TypedDict):
    pred: Union[Tensor, Proxy]
    
class PIDNetModelOutput(ModelOutput):
    extra_p: Optional[Union[Tensor, Proxy]]
    extra_d: Optional[Union[Tensor, Proxy]]

class SeparateForwardModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def forward_training(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def forward_inference(self, *args, **kwargs):
        raise NotImplementedError
    
    # @final
    def forward(self, *args, **kwargs):
        # TODO: train/val/infer
        return self.forward_training(*args, **kwargs)
        # return self.forward_inference(*args, **kwargs)