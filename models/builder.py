import importlib
from pathlib import Path
import torch
import torch.nn as nn
import models.backbones as backbones
import models.heads as heads

PRETRAINED_ROOT = "pretrained"
UPDATE_PREFIX = "updated_"


def load_pretrained_checkpoint(model_name: str):
    supporting_model_list = ["atomixnet_l", "atomixnet_m", "atomixnet_s", "resnet50", "segformer"]
    assert model_name in supporting_model_list

    pretrained_root = Path(PRETRAINED_ROOT)

    if model_name == "atomixnet_l":
        state_dict = torch.load(pretrained_root / "backbones" / "atomixnet" / "atomixnet_l.pth")
    elif model_name == "atomixnet_m":
        state_dict = torch.load(pretrained_root / "backbones" / "atomixnet" / "atomixnet_m.pth")
    elif model_name == "atomixnet_s":
        state_dict = torch.load(pretrained_root / "backbones" / "atomixnet" / "atomixnet_s.pth")
    elif model_name == "resnet50":
        state_dict = torch.load(pretrained_root / "backbones" / "resnet" / "resnet50.pth")
    elif model_name == "segformer":
        state_dict = torch.load(pretrained_root / "backbones" / "segformer" / "segformer.pth")
    else:
        raise AssertionError(f"The model_name ({model_name}) should be one of the followings: {supporting_model_list}")

    return state_dict


class AssembleModel(nn.Module):
    def __init__(self, args, num_classes) -> None:
        super(AssembleModel, self).__init__()
        self.task = args.train.task
        backbone_name = args.train.architecture.backbone
        head = args.train.architecture.head

        self.backbone: nn.Module = eval(f"backbones.{backbone_name}")()
        model_state_dict = load_pretrained_checkpoint(backbone_name)
        self.backbone.load_state_dict(model_state_dict)
        self._freeze_backbone()

        if self.task == 'classification':
            head_module: nn.Module = eval(f"heads.{self.task}.{head}")
            self.head = head_module(feature_dim=self.backbone.last_channels, num_classes=num_classes)
        if self.task == 'segmentation':
            head_module: nn.Module = eval(f"heads.{self.task}.{head}")
            self.head = head_module(feature_dim=self.backbone.last_channels, num_classes=num_classes)

    def _freeze_backbone(self):
        for m in self.backbone.parameters():
            m.requires_grad = False

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
