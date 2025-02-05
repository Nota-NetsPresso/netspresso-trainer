# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import argparse
import os
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from netspresso_trainer.models import build_model
from netspresso_trainer.models.utils import is_single_task_model
from netspresso_trainer.utils.onnx import save_onnx
from omegaconf import OmegaConf

TEMP_NUM_CLASSES = 80

def parse_args():

    parser = argparse.ArgumentParser(description="Parser for NetsPresso fx tracing checker")

    parser.add_argument(
        '-c', '--config-path', type=str, default="config/model/yolox/yolox-s-detection.yaml",
        help="Model config path")
    parser.add_argument(
        '-o', '--output-dir', type=str, default="onnx/",
        help="ONNX model output path")
    parser.add_argument(
        '--sample-size', type=list, default=[640, 640],
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
    os.makedirs(args.output_dir, exist_ok=True)

    for model_config_path in config_path_list:
        try:
            print(f"ONNX export for ({model_config_path})..... ", end='', flush=True)
            config = OmegaConf.load(model_config_path)
            config = config.model
            config.single_task_model = is_single_task_model(config)
            torch_model: nn.Module = build_model(config, num_classes=TEMP_NUM_CLASSES, devices=torch.device("cpu"), distributed=False)
            torch_model.eval()
            print(torch_model)
            save_onnx(torch_model, f=Path(args.output_dir) / f"{model_config_path.stem}.onnx",
                      sample_input=torch.randn(1, 3, *args.sample_size), opset_version=17)
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
