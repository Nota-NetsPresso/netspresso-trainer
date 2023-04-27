from pathlib import Path

from omegaconf import OmegaConf
import torch

from models.backbones.experimental.vit import vit
from utils.pretrained_editor import convert_state_dict_to_model

model = vit()

checkpoint_path = Path("pretrained") / "backbones_backup" / "vit" / "vit-tiny.pt"
state_dict = torch.load(str(checkpoint_path))
convert_state_dict_to_model('vit', model, state_dict)