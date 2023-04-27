"""
Based on the vit implementation of apple/ml-cvnets.
https://github.com/apple/ml-cvnets/blob/84d992f413e52c0468f86d23196efd9dad885e6f/cvnets/models/classification/vit.py
"""

from typing import Dict

from omegaconf import OmegaConf


def get_configuration() -> Dict:
    opts = OmegaConf.load("models/configuration/vit.yaml")
    mode = getattr(opts, "model.classification.vit.mode", "tiny")
    # if not mode:
    #     logger.error("Please specify mode")

    mode = mode.lower()
    dropout = getattr(opts, "model.classification.vit.dropout", 0.0)
    norm_layer = getattr(opts, "model.classification.vit.norm_layer", "layer_norm")

    vit_config = dict()
    if mode == "tiny":
        vit_config = {
            "patch_size": 16,
            "embed_dim": 192,
            "n_transformer_layers": 12,
            "n_attn_heads": 3,
            "ffn_dim": 192 * 4,
            "norm_layer": "layer_norm",
            "pos_emb_drop_p": 0.1,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "small":
        vit_config = {
            "patch_size": 16,
            "embed_dim": 384,
            "n_transformer_layers": 12,
            "n_attn_heads": 6,
            "ffn_dim": 384 * 4,
            "norm_layer": "layer_norm",
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    # else:
    #     logger.error("Got unsupported ViT configuration: {}".format(mode))
    return vit_config
