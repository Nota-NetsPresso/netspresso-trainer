import argparse
from collections import deque

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_path", type=str, required=True, help="Path to the input model")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to save the output model")
    args = parser.parse_args()

    assert args.input_model_path.endwith('.pt'), "Input model must be a .pt file (torch.fx)"
    assert args.output_model_path.endwith('.pt'), "Output model must be a .pt file (torch.fx)"

    # Load the model
    model = torch.load(args.input_model_path)

    # Remove the segmentation head for auxiliary loss
    output_node = list(model.graph.nodes)[-1]
    model_output = output_node.args[0]
    remove_target_nodes = deque([model_output['extra_p'], model_output['extra_d']])
    model_output = {'pred': model_output['pred']} # Only keep the 'pred' key
    output_node.args = (model_output,)

    # Remove the extra nodes
    while remove_target_nodes:
        node = remove_target_nodes.popleft()
        if not bool(node.users):
            remove_target_nodes.extend(node.args)
            model.graph.erase_node(node)
    model.graph.eliminate_dead_code()
    model.recompile()

    # Save the modified model
    torch.save(model, args.output_model_path)
