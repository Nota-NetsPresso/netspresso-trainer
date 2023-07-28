from pathlib import Path

from omegaconf import OmegaConf
import torch

from models.backbones.experimental.vit import vit
from utils.pretrained_editor import convert_state_dict_to_model

yaml_path = Path("models/card") / "vit.yaml"

model = vit(task='classification')
# print(list(model.state_dict().keys()))

checkpoint_path = Path("/CHECKPOINT") / "backbones_backup" / "vit" / "vit-tiny.pt"
state_dict = torch.load(str(checkpoint_path))
# print(list(state_dict.keys()))

convert_state_dict_to_model(yaml_path,
                            model=model,
                            state_dict=state_dict)
