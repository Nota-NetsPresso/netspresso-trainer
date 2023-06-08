from pathlib import Path

from omegaconf import OmegaConf
import torch

from models.backbones.experimental.resnet import resnet50
from utils.pretrained_editor import convert_state_dict_to_model

yaml_path = Path("models/card") / "resnet50.yaml"

model = resnet50(task='classification')

checkpoint_path = Path("pretrained") / "backbones_backup" / "resnet" / "resnet50.pth"
state_dict = torch.load(str(checkpoint_path))

convert_state_dict_to_model(yaml_path,
                            model=model,
                            state_dict=state_dict)
