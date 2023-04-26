import importlib
from pathlib import Path
import torch
import torch.nn as nn
import models.full as full
import models.backbones as backbones
import models.heads as heads

PRETRAINED_ROOT = Path("pretrained")
UPDATE_PREFIX = "updated_"

MODEL_PRETRAINED_DICT = {
    "atomixnet_l": PRETRAINED_ROOT / "backbones" / "atomixnet" / "atomixnet_l.pth",
    "atomixnet_m": PRETRAINED_ROOT / "backbones" / "atomixnet" / "atomixnet_m.pth",
    "atomixnet_s": PRETRAINED_ROOT / "backbones" / "atomixnet" / "atomixnet_s.pth",
    "resnet50": PRETRAINED_ROOT / "backbones" / "resnet" / "resnet50.pth",
    "segformer": PRETRAINED_ROOT / "backbones" / "segformer" / "segformer.pth",
    "pidnet": PRETRAINED_ROOT / "full" / "pidnet" / "pidnet_s.pth",
}

def load_pretrained_checkpoint(model_name: str):
    supporting_model_list = ["atomixnet_l", "atomixnet_m", "atomixnet_s", "resnet50", "segformer", "pidnet"]
    assert model_name in supporting_model_list

    if model_name in MODEL_PRETRAINED_DICT:
        state_dict_path: Path = MODEL_PRETRAINED_DICT[model_name]
        state_dict = torch.load(state_dict_path)
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
        try:
            model_state_dict = load_pretrained_checkpoint(backbone_name)
            self.backbone.load_state_dict(model_state_dict)
        except AssertionError as e:
            pass
        # self._freeze_backbone()

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
        model_name = args.train.architecture.full
        model: nn.Module = eval(f"full.{model_name}")(args, num_classes)
        
        model_state_dict = load_pretrained_checkpoint(model_name)
        model.load_state_dict(model_state_dict, strict=False)
        return model

    model = AssembleModel(args, num_classes)
    return model
