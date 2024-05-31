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

from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import save_file
from omegaconf import ListConfig, OmegaConf


def split_qkv(dest_layer_list: List[str], model_state_dict: Dict[str, torch.Tensor], layer_value: torch.Tensor, layer_name: str):
    assert "qkv" in layer_name, f"[{layer_name}] may be not qkv projection layer..."
    out_original = layer_value.size(0)  # (C_qk + C_qk + C_v), ... (if bias, dim=1)
    print(layer_name, layer_value.size())
    
    out_dict: Dict[str, torch.Tensor] = {}
    channel_start_idx = 0
    for dest_layer in dest_layer_list:
        assert dest_layer in model_state_dict, f"{dest_layer} not in model's state dict."
        dest_weight_tensor = model_state_dict[dest_layer]

        out_dest = dest_weight_tensor.size(0)  # (C_qk or C_v), ... (if bias, dim=1)
        print(f"{channel_start_idx}:{channel_start_idx+out_dest}, ...", layer_value.size(), dest_layer, dest_weight_tensor.size())
        
        if layer_value.dim() != 1 and dest_weight_tensor.dim() != 1:  # if not bias, assertion check with sequence length(s)
            in_dest, in_original = layer_value.size(1), dest_weight_tensor.size(1)
            assert in_dest == in_original
        
        out_dict[dest_layer] = layer_value[channel_start_idx:channel_start_idx+out_dest, ...]
        channel_start_idx += out_dest  # cumulation of channel start idx
    
    assert channel_start_idx == out_original
    return out_dict
    

def convert_state_dict_to_model(yaml_path, model, state_dict):

    config = OmegaConf.load(yaml_path)
    mapping = config.mapping

    model_state_dict = model.state_dict()
    if config.pretrained.state_dict_key is not None:
        state_dict_key = config.pretrained.state_dict_key
        state_dict = state_dict[state_dict_key]

    print(list(model_state_dict.keys())[:5])
    print(list(state_dict.keys())[:5])

    extracted_state_dict = {}
    for layer_name, value in state_dict.items():
        assert layer_name in mapping, f"{layer_name} not in {yaml_path}"
        dest_layer_name = mapping[layer_name]
        
        if dest_layer_name is None:
            print(f"{layer_name} -> None. Skipped.")
            continue

        if isinstance(dest_layer_name, ListConfig):
            dest_layer_list = list(dest_layer_name)
            dest_layer_dict = split_qkv(dest_layer_list, model_state_dict, value, layer_name)
            extracted_state_dict.update(dest_layer_dict)
            continue

        assert dest_layer_name in model_state_dict, f"{dest_layer_name} ({type(dest_layer_name)}) not in model_state_dict"
        extracted_state_dict[dest_layer_name] = value

    no_match_layers = set(model_state_dict).difference(set(extracted_state_dict))
    no_match_layers = sorted(no_match_layers)

    print(f"no_match_layers: \n{no_match_layers[:]}")
    print(f"NO MATCH COUNT: {len(no_match_layers)}")
    model.load_state_dict(dict(extracted_state_dict), strict=False)
    _save_extracted_state_dict(extracted_state_dict, "result.safetensors")

    print("Complete!")
    return model


def _save_extracted_state_dict(extracted_state_dict, f):
    save_file(extracted_state_dict, f)


if __name__ == '__main__':
    from netspresso_trainer.models.backbones.experimental.efficientformer import efficientformer

    yaml_path = Path("models/card") / "efficientformer.yaml"

    model = efficientformer(task='classification')
    # print(list(model.state_dict().keys()))

    checkpoint_path = Path("/CHECKPOINT") / "backbones_backup" / "efficientformer" / "efficientformer_l1_1000d.pth"
    state_dict = torch.load(str(checkpoint_path))
    # print(list(state_dict.keys()))

    convert_state_dict_to_model(yaml_path, model=model, state_dict=state_dict)
