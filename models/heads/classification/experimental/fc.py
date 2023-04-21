import torch
import torch.nn as nn

from models.utils import SeparateForwardModule

class FC(SeparateForwardModule):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super(FC, self).__init__()
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward_training(self, x):
        x = self.classifier(x)
        return {'pred': x}
    
    def forward_inference(self, x):
        return self.forward_training(x)['pred']
    
def fc(feature_dim, num_classes):
    return FC(feature_dim=feature_dim, num_classes=num_classes)