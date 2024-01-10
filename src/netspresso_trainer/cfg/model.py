from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import MISSING, MissingMandatoryValue

__all__ = [
    "ModelConfig",
    "ClassificationEfficientFormerModelConfig",
    "SegmentationEfficientFormerModelConfig",
    "DetectionEfficientFormerModelConfig",
    "ClassificationMobileNetV3ModelConfig",
    "SegmentationMobileNetV3ModelConfig",
    "DetectionMobileNetV3ModelConfig",
    "ClassificationMobileViTModelConfig",
    "PIDNetModelConfig",
    "ClassificationResNetModelConfig",
    "SegmentationResNetModelConfig",
    "DetectionResNetModelConfig",
    "SegmentationSegFormerModelConfig",
    "ClassificationViTModelConfig",
    "DetectionYoloXModelConfig",
    "ClassificationMixNetSmallModelConfig",
    "ClassificationMixNetMediumModelConfig",
    "ClassificationMixNetLargeModelConfig",
    "SegmentationMixNetSmallModelConfig",
    "SegmentationMixNetMediumModelConfig",
    "SegmentationMixNetLargeModelConfig",
    "DetectionMixNetSmallModelConfig",
    "DetectionMixNetMediumModelConfig",
    "DetectionMixNetLargeModelConfig",
]


@dataclass
class ArchitectureConfig:
    full: Optional[Dict[str, Any]] = None
    backbone: Optional[Dict[str, Any]] = None
    neck: Optional[Dict[str, Any]] = None
    head: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        assert bool(self.full) != bool(self.backbone), "Only one of full or backbone should be given."


@dataclass
class ModelConfig:
    task: str = MISSING
    name: str = MISSING
    checkpoint: Optional[Union[Path, str]] = None
    load_checkpoint_head: bool = False
    fx_model_checkpoint: Optional[Union[Path, str]] = None
    resume_optimizer_checkpoint: Optional[Union[Path, str]] = None
    freeze_backbone: bool = False
    architecture: ArchitectureConfig = field(default_factory=lambda: ArchitectureConfig())
    losses: Optional[List[Dict[str, Any]]] = None


@dataclass
class EfficientFormerArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "efficientformer",
        "params": {
            "num_attention_heads": 8,
            "attention_channels": 256,
            "attention_dropout_prob": 0.,
            "attention_value_expansion_ratio": 4,
            "ffn_intermediate_ratio": 4,
            "ffn_dropout_prob": 0.,
            "ffn_act_type": 'gelu',
            "vit_num": 1,
        },
        "stage_params": [
            {"num_blocks": 3, "channels": 48},
            {"num_blocks": 2, "channels": 96},
            {"num_blocks": 6, "channels": 224},
            {"num_blocks": 4, "channels": 448},
        ],
    })


@dataclass
class MobileNetV3ArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mobilenetv3",
        "params": None,
        "stage_params": [
            {
                "in_channels": [16],
                "kernel_sizes": [3],
                "expanded_channels": [16],
                "out_channels": [16],
                "use_se": [True],
                "act_type": ["relu"],
                "stride": [2],
            },
            {
                "in_channels": [16, 24],
                "kernel_sizes": [3, 3],
                "expanded_channels": [72, 88],
                "out_channels": [24, 24],
                "use_se": [False, False],
                "act_type": ["relu", "relu"],
                "stride": [2, 1],
            },
            {
                "in_channels": [24, 40, 40, 40, 48],
                "kernel_sizes": [5, 5, 5, 5, 5],
                "expanded_channels": [96, 240, 240, 120, 144],
                "out_channels": [40, 40, 40, 48, 48],
                "use_se": [True, True, True, True, True],
                "act_type": ["hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish"],
                "stride": [2, 1, 1, 1, 1],
            },
            {
                "in_channels": [48, 96, 96],
                "kernel_sizes": [5, 5, 5],
                "expanded_channels": [288, 576, 576],
                "out_channels": [96, 96, 96],
                "use_se": [True, True, True],
                "act_type": ["hard_swish", "hard_swish", "hard_swish"],
                "stride": [2, 1, 1],
            },
        ],
    })


@dataclass
class MobileViTArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mobilevit",
        "params": {
            "patch_size": 2,
            "num_attention_heads": 4,
            "attention_dropout_prob": 0.1,
            "ffn_dropout_prob": 0.0,
            "output_expansion_ratio": 4,
            "use_fusion_layer": True,
        },
        "stage_params": [
            {
                "out_channels": 32,
                "block_type": "mv2",
                "num_blocks": 1,
                "stride": 1,
                "ir_expansion_ratio": 4,
            },
            {
                "out_channels": 64,
                "block_type": "mv2",
                "num_blocks": 3,
                "stride": 2,
                "ir_expansion_ratio": 4,
            },
            {
                "out_channels": 96,
                "block_type": "mobilevit",
                "num_blocks": 2,
                "stride": 2,
                "hidden_size": 144,
                "intermediate_size": 288,
                "dilate": False,
                "ir_expansion_ratio": 4,
            },
            {
                "out_channels": 128,
                "block_type": "mobilevit",
                "num_blocks": 4,
                "stride": 2,
                "hidden_size": 192,
                "intermediate_size": 384,
                "dilate": False,
                "ir_expansion_ratio": 4,
            },
            {
                "out_channels": 160,
                "block_type": "mobilevit",
                "num_blocks": 3,
                "stride": 2,
                "hidden_size": 240,
                "intermediate_size": 480,
                "dilate": False,
                "ir_expansion_ratio": 4,
            },
        ]
    })


@dataclass
class PIDNetArchitectureConfig(ArchitectureConfig):
    full: Dict[str, Any] = field(default_factory=lambda: {
        "name": "pidnet",
        "m": 2,
        "n": 3,
        "channels": 32,
        "ppm_channels": 96,
        "head_channels": 128,
    })


@dataclass
class ResNetArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "resnet",
        "params": {
            "block_type": "bottleneck",
            "norm_type": "batch_norm",
        },
        "stage_params": [
            {"channels": 64, "num_blocks": 3},
            {"channels": 128, "num_blocks": 4, "replace_stride_with_dilation": False},
            {"channels": 256, "num_blocks": 6, "replace_stride_with_dilation": False},
            {"channels": 512, "num_blocks": 3, "replace_stride_with_dilation": False},
        ],
    })


@dataclass
class SegFormerArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mixtransformer",
        "params": {
            "ffn_intermediate_expansion_ratio": 4,
            "ffn_act_type": "gelu",
            "ffn_dropout_prob": 0.0,
            "attention_dropout_prob": 0.0,
        },
        "stage_params": [
            {
                "num_blocks": 2,
                "sequence_reduction_ratio": 8,
                "attention_chananels": 32,
                "embedding_patch_sizes": 7,
                "embedding_strides": 4,
                "num_attention_heads": 1,
            },
            {
                "num_blocks": 2,
                "sequence_reduction_ratio": 4,
                "attention_chananels": 64,
                "embedding_patch_sizes": 3,
                "num_attention_heads": 2,
            },
            {
                "num_blocks": 2,
                "sequence_reduction_ratio": 2,
                "attention_chananels": 160,
                "embedding_patch_sizes": 3,
                "embedding_strides": 2,
                "num_attention_heads": 5,
            },
            {
                "num_blocks": 2,
                "sequence_reduction_ratio": 1,
                "attention_chananels": 256,
                "embedding_patch_sizes": 3,
                "embedding_strides": 2,
                "num_attention_heads": 8,
            },
        ],
    })


@dataclass
class ViTArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "vit",
        "params": {
            "patch_size": 16,
            "attention_channels": 192,
            "num_blocks": 12,
            "num_attention_heads": 3,
            "attention_dropout_prob": 0.0,
            "ffn_intermediate_channels": 768,
            "ffn_dropout_prob": 0.1,
            "use_cls_token": True,
            "vocab_size": 1000,
        },
        "stage_params": None,
    })


@dataclass
class MixNetSmallArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mixnet",
        "params": {
            "stem_channels": 16,
            "wid_mul": 1.0,
            "dep_mul": 1.0,
            "dropout_rate": 0.,
        },
        "stage_params":  [
            {
                "expansion_ratio": [1, 6, 3],
                "out_channels": [16, 24, 24],
                "num_blocks": [1, 1, 1],
                "kernel_sizes": [[3], [3], [3]],
                "num_exp_groups": [1, 2, 2],
                "num_poi_groups": [1, 2, 2],
                "stride": [1, 2, 1],
                "act_type": ["relu", "relu", "relu"],
                "se_reduction_ratio": [None, None, None],
            },
            {
                "expansion_ratio": [6, 6],
                "out_channels": [40, 40],
                "num_blocks": [1, 3],
                "kernel_sizes": [[3, 5, 7], [3, 5]],
                "num_exp_groups": [1, 2],
                "num_poi_groups": [1, 2],
                "stride": [2, 1],
                "act_type": ["swish", "swish"],
                "se_reduction_ratio": [2, 2],
            },
            {
                "expansion_ratio": [6, 6, 6, 3],
                "out_channels": [80, 80, 120, 120],
                "num_blocks": [1, 2, 1, 2],
                "kernel_sizes": [[3, 5, 7], [3, 5], [3, 5, 7], [3, 5, 7, 9]],
                "num_exp_groups": [1, 1, 2, 2],
                "num_poi_groups": [2, 2, 2, 2],
                "stride": [2, 1, 1, 1],
                "act_type": ["swish", "swish", "swish", "swish"],
                "se_reduction_ratio": [4, 4, 2, 2],
            },
            {
                "expansion_ratio": [6, 6],
                "out_channels": [200, 200],
                "num_blocks": [1, 2],
                "kernel_sizes": [[3, 5, 7, 9, 11], [3, 5, 7, 9]],
                "num_exp_groups": [1, 1],
                "num_poi_groups": [1, 2],
                "stride": [2, 1],
                "act_type": ["swish", "swish"],
                "se_reduction_ratio": [2, 2],
            },
        ],
    })


@dataclass
class MixNetMediumArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mixnet",
        "params": {
            "stem_channels": 24,
            "wid_mul": 1.0,
            "dep_mul": 1.0,
            "dropout_rate": 0.,
        },
        "stage_params":  [
            {
                "expansion_ratio": [1, 6, 3],
                "out_channels": [24, 32, 32],
                "num_blocks": [1, 1, 1],
                "kernel_sizes": [[3], [3, 5, 7], [3]],
                "num_exp_groups": [1, 2, 2],
                "num_poi_groups": [1, 2, 2],
                "stride": [1, 2, 1],
                "act_type": ["relu", "relu", "relu"],
                "se_reduction_ratio": [None, None, None],
            },
            {
                "expansion_ratio": [6, 6],
                "out_channels": [40, 40],
                "num_blocks": [1, 3],
                "kernel_sizes": [[3, 5, 7, 9], [3, 5]],
                "num_exp_groups": [1, 2],
                "num_poi_groups": [1, 2],
                "stride": [2, 1],
                "act_type": ["swish", "swish"],
                "se_reduction_ratio": [2, 2],
            },
            {
                "expansion_ratio": [6, 6, 6, 3],
                "out_channels": [80, 80, 120, 120],
                "num_blocks": [1, 3, 1, 3],
                "kernel_sizes": [[3, 5, 7], [3, 5, 7, 9], [3], [3, 5, 7, 9]],
                "num_exp_groups": [1, 2, 1, 2],
                "num_poi_groups": [1, 2, 1, 2],
                "stride": [2, 1, 1, 1],
                "act_type": ["swish", "swish", "swish", "swish"],
                "se_reduction_ratio": [4, 4, 2, 2],
            },
            {
                "expansion_ratio": [6, 6],
                "out_channels": [200, 200],
                "num_blocks": [1, 3],
                "kernel_sizes": [[3, 5, 7, 9], [3, 5, 7, 9]],
                "num_exp_groups": [1, 1],
                "num_poi_groups": [1, 2],
                "stride": [2, 1],
                "act_type": ["swish", "swish"],
                "se_reduction_ratio": [2, 2],
            },
        ],
    })


@dataclass
class MixNetLargeArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mixnet",
        "params": {
            "stem_channels": 24,
            "wid_mul": 1.3,
            "dep_mul": 1.0,
            "dropout_rate": 0.,
        },
        "stage_params":  [
            {
                "expansion_ratio": [1, 6, 3],
                "out_channels": [24, 32, 32],
                "num_blocks": [1, 1, 1],
                "kernel_sizes": [[3], [3, 5, 7], [3]],
                "num_exp_groups": [1, 2, 2],
                "num_poi_groups": [1, 2, 2],
                "stride": [1, 2, 1],
                "act_type": ["relu", "relu", "relu"],
                "se_reduction_ratio": [None, None, None],
            },
            {
                "expansion_ratio": [6, 6],
                "out_channels": [40, 40],
                "num_blocks": [1, 3],
                "kernel_sizes": [[3, 5, 7, 9], [3, 5]],
                "num_exp_groups": [1, 2],
                "num_poi_groups": [1, 2],
                "stride": [2, 1],
                "act_type": ["swish", "swish"],
                "se_reduction_ratio": [2, 2],
            },
            {
                "expansion_ratio": [6, 6, 6, 3],
                "out_channels": [80, 80, 120, 120],
                "num_blocks": [1, 3, 1, 3],
                "kernel_sizes": [[3, 5, 7], [3, 5, 7, 9], [3], [3, 5, 7, 9]],
                "num_exp_groups": [1, 2, 1, 2],
                "num_poi_groups": [1, 2, 1, 2],
                "stride": [2, 1, 1, 1],
                "act_type": ["swish", "swish", "swish", "swish"],
                "se_reduction_ratio": [4, 4, 2, 2],
            },
            {
                "expansion_ratio": [6, 6],
                "out_channels": [200, 200],
                "num_blocks": [1, 3],
                "kernel_sizes": [[3, 5, 7, 9], [3, 5, 7, 9]],
                "num_exp_groups": [1, 1],
                "num_poi_groups": [1, 2],
                "stride": [2, 1],
                "act_type": ["swish", "swish"],
                "se_reduction_ratio": [2, 2],
            },
        ],
    })


@dataclass
class CSPDarkNetSmallArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "cspdarknet",
        "params": {
            "dep_mul": 0.33,
            "wid_mul": 0.5,
            "act_type": "silu",
        },
        "stage_params": None,
    })


@dataclass
class ClassificationEfficientFormerModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "efficientformer_l1"
    checkpoint: Optional[Union[Path, str]] = "./weights/efficientformer/efficientformer_l1_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: EfficientFormerArchitectureConfig(
        head={
            "name": "fc",
            "params": {
                "intermediate_channels": 1024,
                "num_layers": 1,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationEfficientFormerModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "efficientformer_l1"
    checkpoint: Optional[Union[Path, str]] = "./weights/efficientformer/efficientformer_l1_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: EfficientFormerArchitectureConfig(
        head={
            "name": "all_mlp_decoder",
            "params": {
                "intermediate_channels": 256,
                "classifier_dropout_prob": 0.,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class DetectionEfficientFormerModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "efficientformer_l1"
    checkpoint: Optional[Union[Path, str]] = "./weights/efficientformer/efficientformer_l1_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: EfficientFormerArchitectureConfig(
        neck={
            "name": "fpn",
            "params": {
                "num_outs": 4,
                "start_level": 0,
                "end_level": -1,
                "add_extra_convs": False,
                "relu_before_extra_convs": False,
            },
        },
        head={
            "name": "anchor_decoupled_head",
            "params": {
                # Anchor parameters
                "anchor_sizes": [[32,], [64,], [128,], [256,]],
                "aspect_ratios": [0.5, 1.0, 2.0],
                "num_layers": 1,
                "norm_type": "batch_norm",
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "retinanet_loss", "weight": None},
    ])


@dataclass
class ClassificationMobileNetV3ModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mobilenet_v3_small"
    checkpoint: Optional[Union[Path, str]] = "./weights/mobilenetv3/mobilenet_v3_small_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MobileNetV3ArchitectureConfig(
        head={
            "name": "fc",
            "params": {
                "intermediate_channels": 1024,
                "num_layers": 1,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationMobileNetV3ModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "mobilenet_v3_small"
    checkpoint: Optional[Union[Path, str]] = "./weights/mobilenetv3/mobilenet_v3_small_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MobileNetV3ArchitectureConfig(
        head={
            "name": "all_mlp_decoder",
            "params": {
                "intermediate_channels": 256,
                "classifier_dropout_prob": 0.,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class DetectionMobileNetV3ModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "mobilenet_v3_small"
    checkpoint: Optional[Union[Path, str]] = "./weights/mobilenetv3/mobilenet_v3_small_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MobileNetV3ArchitectureConfig(
        neck={
            "name": "fpn",
            "params": {
                "num_outs": 4,
                "start_level": 0,
                "end_level": -1,
                "add_extra_convs": False,
                "relu_before_extra_convs": False,
            },
        },
        head={
            "name": "anchor_decoupled_head",
            "params": {
                # Anchor parameters
                "anchor_sizes": [[32,], [64,], [128,], [256,]],
                "aspect_ratios": [0.5, 1.0, 2.0],
                "num_layers": 1,
                "norm_type": "batch_norm",
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "retinanet_loss", "weight": None},
    ])


@dataclass
class ClassificationMobileViTModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mobilevit_s"
    checkpoint: Optional[Union[Path, str]] = "./weights/mobilevit/mobilevit_s_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MobileViTArchitectureConfig(
        head={
            "name": "fc",
            "params": {
                "intermediate_channels": 1024,
                "num_layers": 1,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class PIDNetModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "pidnet_s"
    checkpoint: Optional[Union[Path, str]] = "./weights/pidnet/pidnet_s.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: PIDNetArchitectureConfig())
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "pidnet_loss", "ignore_index": 255, "weight": None},
    ])


@dataclass
class ClassificationResNetModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "resnet50"
    checkpoint: Optional[Union[Path, str]] = "./weights/resnet/resnet50_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: ResNetArchitectureConfig(
        head={
            "name": "fc",
            "params": {
                "intermediate_channels": 1024,
                "num_layers": 1,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationResNetModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "resnet50"
    checkpoint: Optional[Union[Path, str]] = "./weights/resnet/resnet50_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: ResNetArchitectureConfig(
        head={
            "name": "all_mlp_decoder",
            "params": {
                "intermediate_channels": 256,
                "classifier_dropout_prob": 0.,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class DetectionResNetModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "resnet50"
    checkpoint: Optional[Union[Path, str]] = "./weights/resnet/resnet50_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: ResNetArchitectureConfig(
        neck={
            "name": "fpn",
            "params": {
                "num_outs": 4,
                "start_level": 0,
                "end_level": -1,
                "add_extra_convs": False,
                "relu_before_extra_convs": False,
            },
        },
        head={
            "name": "anchor_decoupled_head",
            "params": {
                # Anchor parameters
                "anchor_sizes": [[32,], [64,], [128,], [256,]],
                "aspect_ratios": [0.5, 1.0, 2.0],
                "num_layers": 1,
                "norm_type": "batch_norm",
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "retinanet_loss", "weight": None},
    ])


@dataclass
class SegmentationSegFormerModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "segformer"
    checkpoint: Optional[Union[Path, str]] = "./weights/segformer/segformer_b0.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: SegFormerArchitectureConfig(
        head={
            "name": "all_mlp_decoder",
            "params": {
                "intermediate_channels": 256,
                "classifier_dropout_prob": 0.,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class ClassificationViTModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "vit_tiny"
    checkpoint: Optional[Union[Path, str]] = "./weights/vit/vit_tiny_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: ViTArchitectureConfig(
        head={
            "name": "fc",
            "params": {
                "intermediate_channels": 1024,
                "num_layers": 1,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class DetectionYoloXModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "yolox_s"
    checkpoint: Optional[Union[Path, str]] = "./weights/yolox/yolox_s_coco.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: CSPDarkNetSmallArchitectureConfig(
        neck={
            "name": "yolopafpn",
            "params": {
                "dep_mul": 0.33,
                "act_type": "silu",
            },
        },
        head={
            "name": "anchor_free_decoupled_head",
            "params": {
                "act_type": "silu",
                # postprocessor - decode
                "score_thresh": 0.7,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "yolox_loss", "weight": None}
    ])


@dataclass
class ClassificationMixNetSmallModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mixnet_s"
    checkpoint: Optional[Union[Path, str]] = "./weights/mixnet/mixnet_s_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MixNetSmallArchitectureConfig(
        head={
            "name": "fc",
            "params": {
                "intermediate_channels": 1024,
                "num_layers": 1,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationMixNetSmallModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "mixnet_s"
    checkpoint: Optional[Union[Path, str]] = "./weights/mixnet/mixnet_s_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MixNetSmallArchitectureConfig(
        head={
            "name": "all_mlp_decoder",
            "params": {
                "intermediate_channels": 256,
                "classifier_dropout_prob": 0.,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class DetectionMixNetSmallModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "mixnet_s"
    checkpoint: Optional[Union[Path, str]] = "./weights/mixnet/mixnet_s_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MixNetSmallArchitectureConfig(
        neck={
            "name": "fpn",
            "params": {
                "num_outs": 4,
                "start_level": 0,
                "end_level": -1,
                "add_extra_convs": False,
                "relu_before_extra_convs": False,
            },
        },
        head={
            "name": "anchor_decoupled_head",
            "params": {
                # Anchor parameters
                "anchor_sizes": [[32,], [64,], [128,], [256,]],
                "aspect_ratios": [0.5, 1.0, 2.0],
                "num_layers": 1,
                "norm_type": "batch_norm",
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "retinanet_loss", "weight": None},
    ])


@dataclass
class ClassificationMixNetMediumModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mixnet_m"
    checkpoint: Optional[Union[Path, str]] = "./weights/mixnet/mixnet_m_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MixNetMediumArchitectureConfig(
        head={
            "name": "fc",
            "params": {
                "intermediate_channels": 1024,
                "num_layers": 1,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationMixNetMediumModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "mixnet_m"
    checkpoint: Optional[Union[Path, str]] = "./weights/mixnet/mixnet_m_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MixNetMediumArchitectureConfig(
        head={
            "name": "all_mlp_decoder",
            "params": {
                "intermediate_channels": 256,
                "classifier_dropout_prob": 0.,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class DetectionMixNetMediumModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "mixnet_m"
    checkpoint: Optional[Union[Path, str]] = "./weights/mixnet/mixnet_m_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MixNetMediumArchitectureConfig(
        neck={
            "name": "fpn",
            "params": {
                "num_outs": 4,
                "start_level": 0,
                "end_level": -1,
                "add_extra_convs": False,
                "relu_before_extra_convs": False,
            },
        },
        head={
            "name": "anchor_decoupled_head",
            "params": {
                # Anchor parameters
                "anchor_sizes": [[32,], [64,], [128,], [256,]],
                "aspect_ratios": [0.5, 1.0, 2.0],
                "num_layers": 1,
                "norm_type": "batch_norm",
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "retinanet_loss", "weight": None},
    ])


@dataclass
class ClassificationMixNetLargeModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mixnet_l"
    checkpoint: Optional[Union[Path, str]] = "./weights/mixnet/mixnet_l_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MixNetLargeArchitectureConfig(
        head={
            "name": "fc",
            "params": {
                "intermediate_channels": 1024,
                "num_layers": 1,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationMixNetLargeModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "mixnet_l"
    checkpoint: Optional[Union[Path, str]] = "./weights/mixnet/mixnet_l_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MixNetLargeArchitectureConfig(
        head={
            "name": "all_mlp_decoder",
            "params": {
                "intermediate_channels": 256,
                "classifier_dropout_prob": 0.,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class DetectionMixNetLargeModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "mixnet_l"
    checkpoint: Optional[Union[Path, str]] = "./weights/mixnet/mixnet_l_imagenet1k.safetensors"
    architecture: ArchitectureConfig = field(default_factory=lambda: MixNetLargeArchitectureConfig(
        neck={
            "name": "fpn",
            "params": {
                "num_outs": 4,
                "start_level": 0,
                "end_level": -1,
                "add_extra_convs": False,
                "relu_before_extra_convs": False,
            },
        },
        head={
            "name": "anchor_decoupled_head",
            "params": {
                # Anchor parameters
                "anchor_sizes": [[32,], [64,], [128,], [256,]],
                "aspect_ratios": [0.5, 1.0, 2.0],
                "num_layers": 1,
                "norm_type": "batch_norm",
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            }
        }
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "retinanet_loss", "weight": None},
    ])
