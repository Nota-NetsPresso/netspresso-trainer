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
    "ClassificationMobileViTModelConfig",
    "PIDNetModelConfig",
    "ClassificationResNetModelConfig",
    "SegmentationResNetModelConfig",
    "ClassificationSegFormerModelConfig",
    "SegmentationSegFormerModelConfig",
    "ClassificationViTModelConfig",
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
            "attention_hidden_size": 256,
            "attention_dropout_prob": 0.,
            "attention_ratio": 4,
            "attention_bias_resolution": 16,
            "pool_size": 3,
            "intermediate_ratio": 4,
            "hidden_dropout_prob": 0.,
            "hidden_activation_type": 'gelu',
            "layer_norm_eps": 1e-5,
            "drop_path_rate": 0.,
            "use_layer_scale": True,
            "layer_scale_init_value": 1e-5,
            "down_patch_size": 3,
            "down_stride": 2,
            "down_pad": 1,
            "vit_num": 1,
        },
        "stage_params": [
            {"num_blocks": 3, "hidden_sizes": 48, "downsamples": True},
            {"num_blocks": 2, "hidden_sizes": 96, "downsamples": True},
            {"num_blocks": 6, "hidden_sizes": 224, "downsamples": True},
            {"num_blocks": 4, "hidden_sizes": 448, "downsamples": True},
        ],
    })


@dataclass
class MobileNetV3ArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mobilenetv3_small",
        "params": None,
        "stage_params": [
            {
                "in_channels": [16],
                "kernel": [3],
                "expanded_channels": [16],
                "out_channels": [16],
                "use_se": [True],
                "activation": ["relu"],
                "stride": [2],
                "dilation": [1],
            },
            {
                "in_channels": [16, 24],
                "kernel": [3, 3],
                "expanded_channels": [72, 88],
                "out_channels": [24, 24],
                "use_se": [False, False],
                "activation": ["relu", "relu"],
                "stride": [2, 1],
                "dilation": [1, 1],
            },
            {
                "in_channels": [24, 40, 40, 40, 48],
                "kernel": [5, 5, 5, 5, 5],
                "expanded_channels": [96, 240, 240, 120, 144],
                "out_channels": [40, 40, 40, 48, 48],
                "use_se": [True, True, True, True, True],
                "activation": ["hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish"],
                "stride": [2, 1, 1, 1, 1],
                "dilation": [1, 1, 1, 1, 1],
            },
            {
                "in_channels": [48, 96, 96],
                "kernel": [5, 5, 5],
                "expanded_channels": [288, 576, 576],
                "out_channels": [96, 96, 96],
                "use_se": [True, True, True],
                "activation": ["hard_swish", "hard_swish", "hard_swish"],
                "stride": [2, 1, 1],
                "dilation": [1, 1, 1],
            },
        ],
    })


@dataclass
class MobileViTArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mobilevit",
        "params": {
            "patch_embedding_out_channels": 16,
            "local_kernel_size": 3,
            "patch_size": 2,
            "num_attention_heads": 4,
            "attention_dropout_prob": 0.1,
            "hidden_dropout_prob": 0.0,
            "exp_factor": 4,
            "layer_norm_eps": 1e-5,
            "use_fusion_layer": True,
        },
        "stage_params": [
            {
                "out_channels": 32,
                "block_type": "mv2",
                "num_blocks": 1,
                "stride": 1,
                "hidden_size": None,
                "intermediate_size": None,
                "num_transformer_blocks": None,
                "dilate": None,
                "expand_ratio": 4,
            },
            {
                "out_channels": 64,
                "block_type": "mv2",
                "num_blocks": 3,
                "stride": 2,
                "hidden_size": None,
                "intermediate_size": None,
                "num_transformer_blocks": None,
                "dilate": None,
                "expand_ratio": 4,
            },
            {
                "out_channels": 96,
                "block_type": "mobilevit",
                "num_blocks": None,
                "stride": 2,
                "hidden_size": 144,
                "intermediate_size": 288,
                "num_transformer_blocks": 2,
                "dilate": False,
                "expand_ratio": 4,
            },
            {
                "out_channels": 128,
                "block_type": "mobilevit",
                "num_blocks": None,
                "stride": 2,
                "hidden_size": 192,
                "intermediate_size": 384,
                "num_transformer_blocks": 4,
                "dilate": False,
                "expand_ratio": 4,
            },
            {
                "out_channels": 160,
                "block_type": "mobilevit",
                "num_blocks": None,
                "stride": 2,
                "hidden_size": 240,
                "intermediate_size": 480,
                "num_transformer_blocks": 3,
                "dilate": False,
                "expand_ratio": 4,
            },
        ]
    })


@dataclass
class PIDNetArchitectureConfig(ArchitectureConfig):
    full: Dict[str, Any] = field(default_factory=lambda: {
        "name": "pidnet",
        "m": 2,
        "n": 3,
        "planes": 32,
        "ppm_planes": 96,
        "head_planes": 128,
    })


@dataclass
class ResNetArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "resnet50",
        "params": {
            "block": "bottleneck",
            "norm_layer": "batch_norm",
            "groups": 1,
            "width_per_group": 64,
            "zero_init_residual": False,
            "expansion": None,
        },
        "stage_params": [
            {"plane": 64, "layers": 3},
            {"plane": 128, "layers": 4},
            {"plane": 256, "layers": 6},
            {"plane": 512, "layers": 3},
        ],
    })


@dataclass
class SegFormerArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "segformer",
        "params": {
            "intermediate_ratio": 4,
            "hidden_activation_type": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_dropout_prob": 0.0,
            "layer_norm_eps": 1e-5,
        },
        "stage_params": [
            {
                "num_blocks": 2,
                "sr_ratios": 8,
                "hidden_sizes": 32,
                "embedding_patch_sizes": 7,
                "embedding_strides": 4,
                "num_attention_heads": 1,
            },
            {
                "num_blocks": 2,
                "sr_ratios": 4,
                "hidden_sizes": 64,
                "embedding_patch_sizes": 3,
                "embedding_strides": 2,
                "num_attention_heads": 2,
            },
            {
                "num_blocks": 2,
                "sr_ratios": 2,
                "hidden_sizes": 160,
                "embedding_patch_sizes": 3,
                "embedding_strides": 2,
                "num_attention_heads": 5,
            },
            {
                "num_blocks": 2,
                "sr_ratios": 1,
                "hidden_sizes": 256,
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
            "hidden_size": 192,
            "num_blocks": 12,
            "num_attention_heads": 3,
            "attention_dropout_prob": 0.0,
            "intermediate_size": 768,
            "hidden_dropout_prob": 0.1,
            "layer_norm_eps": 1e-6,
            "use_cls_token": True,
            "vocab_size": 1000,
        },
        "stage_params": None,
    })


@dataclass
class ClassificationEfficientFormerModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "efficientformer_l1"
    checkpoint: Optional[Union[Path, str]] = "./weights/efficientformer/efficientformer_l1_1000d.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: EfficientFormerArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationEfficientFormerModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "efficientformer_l1"
    checkpoint: Optional[Union[Path, str]] = "./weights/efficientformer/efficientformer_l1_1000d.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: EfficientFormerArchitectureConfig(
        head={"name": "all_mlp_decoder"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class DetectionEfficientFormerModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "efficientformer_l1"
    checkpoint: Optional[Union[Path, str]] = "./weights/efficientformer/efficientformer_l1_1000d.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: EfficientFormerArchitectureConfig(
        neck={"name": "fpn"},
        head={"name": "faster_rcnn"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "roi_head_loss", "weight": None},
        {"criterion": "rpn_loss", "weight": None},
    ])


@dataclass
class ClassificationMobileNetV3ModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mobilenet_v3_small"
    checkpoint: Optional[Union[Path, str]] = "./weights/mobilenetv3/mobilenet_v3_small.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: MobileNetV3ArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationMobileNetV3ModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "mobilenet_v3_small"
    checkpoint: Optional[Union[Path, str]] = "./weights/mobilenetv3/mobilenet_v3_small.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: MobileNetV3ArchitectureConfig(
        head={"name": "all_mlp_decoder"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class ClassificationMobileViTModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mobilevit_s"
    checkpoint: Optional[Union[Path, str]] = "./weights/mobilevit/mobilevit_s.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: MobileViTArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class PIDNetModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "pidnet_s"
    checkpoint: Optional[Union[Path, str]] = "./weights/pidnet/pidnet_s.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: PIDNetArchitectureConfig())
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "pidnet_cross_entropy", "ignore_index": 255, "weight": None},
        {"criterion": "boundary_loss", "weight": 20.0},
        {"criterion": "pidnet_cross_entropy_with_boundary", "ignore_index": 255, "weight": None},
    ])


@dataclass
class ClassificationResNetModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "resnet50"
    checkpoint: Optional[Union[Path, str]] = "./weights/resnet/resnet50.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: ResNetArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationResNetModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "resnet50"
    checkpoint: Optional[Union[Path, str]] = "./weights/resnet/resnet50.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: ResNetArchitectureConfig(
        head={"name": "all_mlp_decoder"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class ClassificationSegFormerModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "segformer"
    checkpoint: Optional[Union[Path, str]] = "./weights/segformer/segformer.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: SegFormerArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationSegFormerModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "segformer"
    checkpoint: Optional[Union[Path, str]] = "./weights/segformer/segformer.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: SegFormerArchitectureConfig(
        head={"name": "all_mlp_decoder"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])


@dataclass
class ClassificationViTModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "vit_tiny"
    checkpoint: Optional[Union[Path, str]] = "./weights/vit/vit-tiny.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: ViTArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}
    ])
