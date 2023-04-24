from abc import abstractmethod
from typing import Any, Callable  # TODO: final in Python 3.8
import torch
import torch.nn as nn


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