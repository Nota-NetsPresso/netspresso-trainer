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
    'mobilenet_v3_large': 'imagenet1k',
    'mobilenet_v4_conv_small': 'imagenet1k',
    'mobilenet_v4_conv_medium': 'imagenet1k',
    'mobilenet_v4_conv_large': 'imagenet1k',
    'mobilenet_v4_hybrid_medium': 'imagenet1k',
    'mobilenet_v4_hybrid_large': 'imagenet1k',
    'segformer_b0': 'ade20k',
    'mobilevit_s': 'imagenet1k',
    'vit_tiny': 'imagenet1k',
    'efficientformer_l1': 'imagenet1k',
    'mixnet_s': 'imagenet1k',
    'mixnet_m': 'imagenet1k',
    'mixnet_l': 'imagenet1k',
    'pidnet_s': 'cityscapes',
    'yolox_nano': 'coco',
    'yolox_tiny': 'coco',
    'yolox_s': 'coco',
    'yolox_m': 'coco',
    'yolox_l': 'coco',
    'yolox_x': 'coco',
    'rtdetr_res18': 'coco',
    'rtdetr_res50': 'coco',
}

MODEL_CHECKPOINT_URL_DICT = {
    'resnet18': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/resnet/resnet18_imagenet1k.safetensors?versionId=mEn38lTgeWB_kkDQOdiF9DfQKrPL2CCy",
    },
    'resnet34': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/resnet/resnet34_imagenet1k.safetensors?versionId=.6Ezbpm8lsRyW.HrrnL6AyywYnsjlarr",
    },
    'resnet50': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/resnet/resnet50_imagenet1k.safetensors?versionId=UF509XgguL1T3IhxSw1HVpfvnhvGGy6J",
    },
    'mobilenet_v3_small': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilenetv3/mobilenet_v3_small_imagenet1k.safetensors?versionId=dPgl9WzLaJLqmuyDxsCoYawItdi0zES8",
    },
    'mobilenet_v3_large': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilenetv3/mobilenet_v3_large_imagenet1k.safetensors?versionId=jPG4LAueBDO5VrFGLQ51_z.iDHa5lOgP",
    },
    'mobilenet_v4_conv_small': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilenetv4/mobilenet_v4_conv_small_imagenet1k.safetensors?versionId=0bpPNyhCNfF5FzHSnXkJFgP8pyU34GKt",
    },
    'mobilenet_v4_conv_medium': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilenetv4/mobilenet_v4_conv_medium_imagenet1k.safetensors?versionId=buTzldKEk8MSWZHehi494KsNMfP3G1Zr",
    },
    'mobilenet_v4_conv_large': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilenetv4/mobilenet_v4_conv_large_imagenet1k.safetensors?versionId=_5D7G_yhUg2YJqwBBgNgUxCHSQPbERSD",
    },
    'mobilenet_v4_hybrid_medium': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilenetv4/mobilenet_v4_hybrid_medium_imagenet1k.safetensors?versionId=5eBSYAwF.HjVeTWTOS.YFeL3f_FIa6Nv",
    },
    'mobilenet_v4_hybrid_large': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilenetv4/mobilenet_v4_hybrid_large_imagenet1k.safetensors?versionId=UAbo2Ag4dMiOPdgO3qYl2ztK4Vn9PnAv",
    },
    'segformer_b0': {
        'ade20k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/segformer/segformer_b0_ade20k.safetensors?versionId=0RRDpZeHb2VvVzFo2jGZN4A4bVQ.k49l",
    },
    'mobilevit_s': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilevit/mobilevit_s_imagenet1k.safetensors?versionId=IvxVWQ.yqTF9tvZr9E2JLyE7_1dBdDB4",
    },
    'vit_tiny': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/vit/vit_tiny_imagenet1k.safetensors?versionId=FJwTnbWvFxnlIK.57wjWCh517kuFpOkF",
    },
    'efficientformer_l1': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/efficientformer/efficientformer_l1_imagenet1k.safetensors?versionId=U5gtiRXNNyBvXOpawPA7lfFil09rq50S",
    },
    'mixnet_s': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mixnet/mixnet_s_imagenet1k.safetensors?versionId=9YlFMjHa__1GYFv1H4l4_d4NqX_5djT5",
    },
    'mixnet_m': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mixnet/mixnet_m_imagenet1k.safetensors?versionId=F11svT5UpQJWGQTv9B..STYj5_P53O1Y",
    },
    'mixnet_l': {
        'imagenet1k': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mixnet/mixnet_l_imagenet1k.safetensors?versionId=nLUZeSGKWRZVldPLc0FCvFCZc6lPPEyB",
    },
    'pidnet_s': {
        'cityscapes': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/pidnet/pidnet_s_cityscapes.safetensors?versionId=lsgtDpiF1yqJpuCLYpruLdR6on0V53r8",
    },
    'yolox_nano': {
        'coco': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/yolox/yolox_nano_coco.safetensors?versionId=JCXugDTwGegx9Kl6Jc5AMJpIkA.WlNVP"
    },
    'yolox_tiny': {
        'coco': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/yolox/yolox_tiny_coco.safetensors?versionId=lJp1bCEToD_6IaL9kRCqcYIwVZ.QQ.1P"
    },
    'yolox_s': {
        'coco': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/yolox/yolox_s_coco.safetensors?versionId=QRLqHKqhv8TSYBrmsQ3M8lCR8w7HEZyA",
    },
    'yolox_m': {
        'coco': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/yolox/yolox_m_coco.safetensors?versionId=xVUySP8xgVTpa6NhCMQpulqmYeRUAhpS",
    },
    'yolox_l': {
        'coco': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/yolox/yolox_l_coco.safetensors?versionId=1GR6YNRu.yUfnjq8hKPgARyZ6YejdxMB",
    },
    'yolox_x': {
        'coco': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/yolox/yolox_x_coco.safetensors?versionId=NWskUEbSGviBWskHQ3P1dQZXnRXOR1WN",
    },
    'rtdetr_res18': {
        'coco': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/rtdetr/rtdetr_res18_coco.safetensors?versionId=uu9v49NI6rQx8wOY6bJbEXUFOG_R9xqH",
    },
    'rtdetr_res50': {
        'coco': "https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/rtdetr/rtdetr_res50_coco.safetensors?versionId=JHmnjY13BEflpnDCYPFJ1c17UwpqDrLQ",
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

    checkpoint_filename = Path(checkpoint_url).name.split('?versionId')[0] # @illian01: Remove specified version id
    model_checkpoint: Path = DEFAULT_CACHE_DIR / model_name / checkpoint_filename
    model_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Safer switch: only extension, user can use the custom name for checkpoint file
    model_checkpoint = model_checkpoint.with_suffix(Path(checkpoint_filename).suffix)
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


def is_single_task_model(model_conf: omegaconf.DictConfig):
    conf_model_architecture_full = model_conf.architecture.full
    if conf_model_architecture_full is None:
        return False
    if conf_model_architecture_full.name is None:
        return False
    return True

def get_model_format(model_conf: omegaconf.DictConfig):
    if not model_conf.checkpoint.path:
        return 'torch'

    model_path = Path(model_conf.checkpoint.path)
    ext = model_path.suffix

    if ext == '.safetensors':
        return 'torch'
    elif ext == '.pt':
        return 'torch.fx'
    elif ext == '.onnx':
        return 'onnx'
    else:
        raise ValueError(f"Unsupported model format: {model_conf.checkpoint.path}")
