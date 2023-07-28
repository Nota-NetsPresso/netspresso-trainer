from pathlib import Path

from omegaconf import OmegaConf

from utils.fx import convert_graphmodule
from models import build_model


if __name__ == '__main__':
    model_yaml_path = Path("config") / "model" / "pidnet.yaml"
    config = OmegaConf.load(model_yaml_path)
    model = build_model(config, num_classes=20)
    fx_model = convert_graphmodule(model)
    print(fx_model)
