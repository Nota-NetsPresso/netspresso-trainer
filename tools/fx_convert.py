import argparse
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch.fx as fx
import torch.nn as nn
from netspresso_trainer.models import build_model
from netspresso_trainer.utils.fx import convert_graphmodule
from omegaconf import OmegaConf

TEMP_NUM_CLASSES = 80

def parse_args():

    parser = argparse.ArgumentParser(description="Parser for NetsPresso fx tracing checker")

    parser.add_argument(
        '-c', '--config-path', type=str, default="config/model",
        help="Model config path")
    parser.add_argument(
        '--debug', action='store_true', help="Debug mode to check with the error message")

    args, _ = parser.parse_known_args()

    return args

def get_model_config_path_list(config_path_or_dir: Path) -> List[Path]:
    if config_path_or_dir.is_dir():
        config_dir = config_path_or_dir
        return sorted(chain(config_dir.glob("*.yaml"), config_dir.glob("*.yml")))
    config_path = config_path_or_dir
    return [config_path]

if __name__ == '__main__':
    args = parse_args()
    
    config_path_list = get_model_config_path_list(Path(args.config_path))        
    
    for model_config_path in config_path_list:
        try:
            print(f"FX Tracing test for ({model_config_path})..... ", end='', flush=True)            
            config = OmegaConf.load(model_config_path)
            torch_model: nn.Module = build_model(config, num_classes=TEMP_NUM_CLASSES,
                                                 model_checkpoint=None, use_pretrained=False)
            fx_model: fx.graph_module.GraphModule = convert_graphmodule(torch_model)
            print("Success!")
        except KeyboardInterrupt:
            print("")
            break
        except Exception as e:
            print("Failed!")
            if args.debug:
                raise e
            print(e)
            pass