from pathlib import Path
import torch

UPDATE_PREFIX = "updated_"


def convert_state_dict_to_model(model_name, model, state_dict):
    """segformer
    def convert_string(x): return f"segformer.encoder.{x}"
    """
    
    """pidnet
    def convert_string(x): return f"{x}"
    def exclude_string(x: str):
        if x.startswith("seghead_p"):
            return True
        if x.startswith("final_layer"):
            return True
        return False
    """

    """vanila
    """
    def convert_string(x): return f"{x}"
    def exclude_string(x): return False

    print(list(model.state_dict().keys())[:5])
    print(list(state_dict.keys())[:5])

    extracted_state_dict = {}
    no_match_layers = []
    for layer_name, value in model.state_dict().items():
        if convert_string(layer_name) in state_dict:
            extracted_state_dict[layer_name] = state_dict[convert_string(layer_name)]
        else:
            no_match_layers.append(layer_name)
            if exclude_string(layer_name):
                continue    
            extracted_state_dict[layer_name] = value

    print(f"no_match_layers: \n{no_match_layers[:5]}")
    print(f"NO MATCH COUNT: {len(no_match_layers)}")
    model.load_state_dict(dict(extracted_state_dict), strict=False)
    _save_extracted_state_dict(extracted_state_dict, "result.pth")

    return model


def _save_extracted_state_dict(extracted_state_dict, f):
    torch.save(extracted_state_dict, f)
