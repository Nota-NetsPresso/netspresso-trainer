import torch
import torch.nn as nn
import models.backbones as backbones
import models.heads as heads
class AssembleModel(nn.Module):
    def __init__(self, args, num_classes) -> None:
        super(AssembleModel, self).__init__()
        task = args.train.task
        bacbkone = args.architecture.backbone
        head = args.architecture.head
        
        self.backbone = eval(f"backbones.{bacbkone}")()
        if task == 'classification':
            head_module = eval(f"heads.{task}.{head}")
            self.head = head_module(feature_dim=self.backbone.out_features, num_classes=num_classes)
            
    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        
        return out
    

def build_model():
    pass