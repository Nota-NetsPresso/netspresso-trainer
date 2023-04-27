from pathlib import Path

from omegaconf import OmegaConf
import torch

from models.backbones.experimental.mobilevit import mobilevit
from utils.pretrained_editor import convert_state_dict_to_model

model = mobilevit()

checkpoint_path = Path("pretrained") / "backbones_backup" / "mobilevit" / "mobilevit_s.pt"
state_dict = torch.load(str(checkpoint_path))
convert_state_dict_to_model('mobilevit', model, state_dict)