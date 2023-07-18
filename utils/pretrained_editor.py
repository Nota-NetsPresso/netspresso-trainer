from pathlib import Path

from omegaconf import OmegaConf, ListConfig
import torch

UPDATE_PREFIX = "updated_"


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

        if isinstance(dest_layer_name, ListConfig):
            dest_layer_name = None  # TODO: split qkv linear weights (and biases)

        if dest_layer_name is None:
            print(f"{layer_name} -> None. Skipped.")
            continue

        assert dest_layer_name in model_state_dict, f"{dest_layer_name} ({type(dest_layer_name)}) not in model_state_dict"
        extracted_state_dict[dest_layer_name] = value

    no_match_layers = set(model_state_dict).difference(set(extracted_state_dict))
    no_match_layers = sorted(list(no_match_layers))

    print(f"no_match_layers: \n{no_match_layers[:]}")
    print(f"NO MATCH COUNT: {len(no_match_layers)}")
    model.load_state_dict(dict(extracted_state_dict), strict=False)
    _save_extracted_state_dict(extracted_state_dict, "result.pth")

    print("Complete!")
    return model


def _save_extracted_state_dict(extracted_state_dict, f):
    torch.save(extracted_state_dict, f)
