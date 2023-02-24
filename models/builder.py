import torch
import torch.nn as nn
import models.backbones as backbones
import models.heads as heads


class AssembleModel(nn.Module):
    def __init__(self, args, num_classes) -> None:
        super(AssembleModel, self).__init__()
        self.task = args.train.task
        bacbkone = args.train.architecture.backbone
        head = args.train.architecture.head

        self.backbone = eval(f"backbones.{bacbkone}")()
        if self.task == 'classification':
            head_module = eval(f"heads.{self.task}.{head}")
            self.head = head_module(feature_dim=self.backbone.last_channels, num_classes=num_classes)
        if self.task == 'segmentation':
            head_module = eval(f"heads.{self.task}.{head}")
            self.head = head_module(feature_dim=self.backbone.last_channels, num_classes=num_classes)

    def forward(self, x, label_size=None):
        features = self.backbone(x)
        if self.task == 'classification':
            out = self.head(features['last_feature'])
        elif self.task == 'segmentation':
            out = self.head(features['intermediate_features'], label_size=label_size)

        return out


def build_model(args, num_classes):
    if args.train.architecture.full is not None:
        model = eval(args.train.architecture.full)(args, num_classes)
        return model

    model = AssembleModel(args, num_classes)
    return model
