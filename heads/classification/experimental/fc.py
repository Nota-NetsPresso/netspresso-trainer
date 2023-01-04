import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
    ) -> None:
        super(FC, self).__init__()
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        x = self.classifier(x)
        return x