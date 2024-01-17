from pathlib import Path
from typing import Any, List, Optional, TypedDict, Union

import omegaconf
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.fx.proxy import Proxy

from ..utils.checkpoint import load_checkpoint

FXTensorType = Union[Tensor, Proxy]
FXTensorListType = Union[List[Tensor], List[Proxy]]

DEFAULT_CACHE_DIR = Path.home() / ".cache" / f"{__name__.split('.')[0]}"  # ~/.cache/netspresso_trainer

DEFAULT_WEIGHT_VERSION_DICT = {
    'resnet18': 'imagenet1k',
    'resnet34': 'imagenet1k',
    'resnet50': 'imagenet1k',
    'mobilenet_v3_small': 'imagenet1k',
    'segformer_b0': 'undefined',
    'mobilevit_s': 'imagenet1k',
    'vit_tiny': 'imagenet1k',
    'efficientformer_l1': 'imagenet1k',
    'mixnet_s': 'imagenet1k',
    'mixnet_m': 'imagenet1k',
    'mixnet_l': 'imagenet1k',
    'pidnet_s': 'cityscapes',
    'yolox_s': 'coco',
}

MODEL_CHECKPOINT_URL_DICT = {
    'resnet18': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/resnet/resnet18_imagenet1k.safetensors",
    },
    'resnet34': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/resnet/resnet34_imagenet1k.safetensors",
    },
    'resnet50': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/resnet/resnet50_imagenet1k.safetensors",
    },
    'mobilenet_v3_small': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilenetv3/mobilenet_v3_small_imagenet1k.safetensors",
    },
    'segformer_b0': {
        'undefined': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/segformer/segformer_b0.safetensors",
    },
    'mobilevit_s': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilevit/mobilevit_s_imagenet1k.safetensors",
    },
    'vit_tiny': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/vit/vit_tiny_imagenet1k.safetensors",
    },
    'efficientformer_l1': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/efficientformer/efficientformer_l1_imagenet1k.safetensors",
    },
    'mixnet_s': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mixnet/mixnet_s_imagenet1k.safetensors",
    },
    'mixnet_m': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mixnet/mixnet_m_imagenet1k.safetensors",
    },
    'mixnet_l': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mixnet/mixnet_l_imagenet1k.safetensors",
    },
    'pidnet_s': {
        'cityscapes': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/pidnet/pidnet_s_cityscapes.safetensors",
    },
    'yolox_s': {
        'coco': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/yolox/yolox_s_coco.safetensors",
    },
}


class BackboneOutput(TypedDict):
    intermediate_features: Optional[FXTensorListType]
    last_feature: Optional[FXTensorType]


class ModelOutput(TypedDict):
    pred: FXTensorType


class AnchorBasedDetectionModelOutput(ModelOutput):
    anchors: FXTensorType
    cls_logits: FXTensorType
    bbox_regression: FXTensorType


class DetectionModelOutput(ModelOutput):
    boxes: Any
    proposals: Any
    anchors: Any
    objectness: Any
    pred_bbox_detlas: Any
    class_logits: Any
    box_regression: Any
    labels: Any
    regression_targets: Any
    post_boxes: Any
    post_scores: Any
    post_labels: Any


class PIDNetModelOutput(ModelOutput):
    extra_p: Optional[FXTensorType]
    extra_d: Optional[FXTensorType]


def download_model_checkpoint(
    model_name: str,
    task: Optional[str] = None,  # TODO: Pretrained weights can be distinguished by task
) -> Path:
    assert model_name in DEFAULT_WEIGHT_VERSION_DICT
    assert model_name in MODEL_CHECKPOINT_URL_DICT

    # TODO: User can select the specific weight version
    checkpoint_weight_version = DEFAULT_WEIGHT_VERSION_DICT[model_name]

    checkpoint_url = MODEL_CHECKPOINT_URL_DICT[model_name][checkpoint_weight_version]

    checkpoint_filename = Path(checkpoint_url).name
    model_checkpoint: Path = DEFAULT_CACHE_DIR / model_name / checkpoint_filename
    model_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Safer switch: only extension, user can use the custom name for checkpoint file
    model_checkpoint = model_checkpoint.with_suffix(Path(checkpoint_url).suffix)
    if not model_checkpoint.exists():
        torch.hub.download_url_to_file(checkpoint_url, model_checkpoint)

    return model_checkpoint


def load_from_checkpoint(
    model: nn.Module,
    model_checkpoint: Optional[Union[str, Path]],
    load_checkpoint_head: bool,
) -> nn.Module:
    model_name = model.name
    task = model.task

    if model_checkpoint is None:
        assert model_name is not None, "When `use_pretrain` is True, model_name should be given."
        assert model_name in MODEL_CHECKPOINT_URL_DICT, \
            f"model_name {model_name} in path {model_checkpoint} is not valid name!"
        model_checkpoint = download_model_checkpoint(model_name, task)
        logger.info(f"Pretrained model for {model_name} is loaded from: {model_checkpoint}")

    model_state_dict = load_checkpoint(model_checkpoint)
    if not load_checkpoint_head:
        logger.info("-"*40)
        logger.info("Head weights are not loaded because model.checkpoint.load_head is set to False")
        head_keys = [key for key in model_state_dict if key.startswith(model.head_list)]
        for key in head_keys:
            del model_state_dict[key]

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

    if not load_checkpoint_head:
        missing_keys = [key for key in missing_keys if not key.startswith(model.head_list)]

    if len(missing_keys) != 0:
        logger.warning(f"Missing key(s) in state_dict: {missing_keys}")
    if len(unexpected_keys) != 0:
        logger.warning(f"Unexpected key(s) in state_dict: {unexpected_keys}")

    return model


def is_single_task_model(conf_model: omegaconf.DictConfig):
    conf_model_architecture_full = conf_model.architecture.full
    if conf_model_architecture_full is None:
        return False
    if conf_model_architecture_full.name is None:
        return False
    return True
