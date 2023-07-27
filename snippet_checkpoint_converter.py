from pathlib import Path

from omegaconf import OmegaConf
import torch

from models.backbones.experimental.efficientformer import efficientformer
from utils.pretrained_editor import convert_state_dict_to_model

yaml_path = Path("models/card") / "efficientformer.yaml"

model = efficientformer(task='classification')

checkpoint_path = Path("/CHECKPOINT") / "backbones_backup" / "efficientformer" / "efficientformer_l1_1000d.pth"
state_dict = torch.load(str(checkpoint_path))

convert_state_dict_to_model(yaml_path,
                            model=model,
                            state_dict=state_dict)
