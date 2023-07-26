from pathlib import Path

from omegaconf import OmegaConf
import torch

from models.backbones.experimental.segformer import segformer
from utils.pretrained_editor import convert_state_dict_to_model

yaml_path = Path("models/card") / "segformer.yaml"

model = segformer(task='classification')

checkpoint_path = Path("/CHECKPOINT") / "backbones_backup" / "segformer" / "nvidia_mit-b0" / "pytorch_model.bin"
state_dict = torch.load(str(checkpoint_path))

convert_state_dict_to_model(yaml_path,
                            model=model,
                            state_dict=state_dict)
