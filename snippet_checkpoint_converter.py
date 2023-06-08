from pathlib import Path

from omegaconf import OmegaConf
import torch

from models.full.experimental.pidnet import pidnet
from utils.pretrained_editor import convert_state_dict_to_model

yaml_path = Path("models/card") / "pidnet_s.yaml"

model = pidnet(args=None, num_classes=10)

checkpoint_path = Path("pretrained") / "full_backup" / "pidnet" / "PIDNet_S_ImageNet.pth.tar"
state_dict = torch.load(str(checkpoint_path))

convert_state_dict_to_model(yaml_path,
                            model=model,
                            state_dict=state_dict)
