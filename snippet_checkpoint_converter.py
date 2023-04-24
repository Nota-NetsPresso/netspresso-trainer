from pathlib import Path

from omegaconf import OmegaConf
import torch

from models.full.experimental.pidnet import pidnet
from utils.pretrained_editor import convert_state_dict_to_model

args = OmegaConf.load("config/pidnet.yaml")
model = pidnet(args, num_classes=3)

checkpoint_path = Path("pretrained") / "full_backup" / "pidnet" / "PIDNet_S_ImageNet.pth.tar"
state_dict = torch.load(str(checkpoint_path))['state_dict']
convert_state_dict_to_model('pidnet', model, state_dict)