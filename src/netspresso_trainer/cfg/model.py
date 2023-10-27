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
    head: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        assert bool(self.full) != bool(self.backbone), "Only one of full or backbone should be given."
    
@dataclass
class ModelConfig:
    task: str = MISSING
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
        "num_blocks": [3, 2, 6, 4],
        "hidden_sizes": [48, 96, 224, 448],
        "num_attention_heads": 8,
        "attention_hidden_size": 256,  # attention_hidden_size_splitted * num_attention_heads
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
        "downsamples": [True, True, True, True],
        "down_patch_size": 3,
        "down_stride": 2,
        "down_pad": 1,
        "vit_num": 1,
    })


@dataclass
class MobileNetV3ArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mobilenetv3_small",
        
        # [in_channels, kernel, expended_channels, out_channels, use_se, activation, stride, dilation]
        "block_info": [
            [
                [16, 3, 16, 16, True, "relu", 2, 1]
            ],
            [
                [16, 3, 72, 24, False, "relu", 2, 1],
                [24, 3, 88, 24, False, "relu", 1, 1]
            ],
            [
                [24, 5, 96, 40, True, "hard_swish", 2, 1],
                [40, 5, 240, 40, True, "hard_swish", 1, 1],
                [40, 5, 240, 40, True, "hard_swish", 1, 1],
                [40, 5, 120, 48, True, "hard_swish", 1, 1],
                [48, 5, 144, 48, True, "hard_swish", 1, 1]
            ],
            [
                [48, 5, 288, 96, True, "hard_swish", 2, 1],
                [96, 5, 576, 96, True, "hard_swish", 1, 1],
                [96, 5, 576, 96, True, "hard_swish", 1, 1]
            ]
        ]
    })


@dataclass
class MobileViTArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mobilevit",
        "out_channels": [32, 64, 96, 128, 160],
        "block_type": ['mv2', 'mv2', 'mobilevit', 'mobilevit', 'mobilevit'],
        "num_blocks": [1, 3, None, None, None],
        "stride": [1, 2, 2, 2, 2],
        "hidden_size": [None, None, 144, 192, 240],
        "intermediate_size": [None, None, 288, 384, 480],
        "num_transformer_blocks": [None, None, 2, 4, 3],
        "dilate": [None, None, False, False, False],
        "expand_ratio": [4, 4, 4, 4, 4],  # [mv2_exp_mult] * 4
        "patch_embedding_out_channels": 16,
        "local_kernel_size": 3,
        "patch_size": 2,
        "num_attention_heads": 4,  # num_heads
        "attention_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.0,
        "exp_factor": 4,
        "layer_norm_eps": 1e-5,
        "use_fusion_layer": True,
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
        "block": "bottleneck",
        "layers": [3, 4, 6, 3],
    })


@dataclass
class SegFormerArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "segformer",
        "num_modules": 4,
        "num_blocks": [2, 2, 2, 2],
        "sr_ratios": [8, 4, 2, 1],
        "hidden_sizes": [32, 64, 160, 256],
        "embedding_patch_sizes": [7, 3, 3, 3],
        "embedding_strides": [4, 2, 2, 2],
        "num_attention_heads": [1, 2, 5, 8],
        "intermediate_ratio": 4,
        "hidden_activation_type": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_dropout_prob": 0.0,
        "layer_norm_eps": 1e-5,
    })


@dataclass
class ViTArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "vit",
        "patch_size": 16,
        "hidden_size": 192,
        "num_blocks": 12,
        "num_attention_heads": 3,
        "attention_dropout_prob": 0.0,
        "intermediate_size": 192 * 4,
        "hidden_dropout_prob": 0.1,
    })


@dataclass
class ClassificationEfficientFormerModelConfig(ModelConfig):
    task: str = "classification"
    checkpoint: Optional[Union[Path, str]] = "./weights/efficientformer/efficientformer_l1_1000d.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: EfficientFormerArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "label_smoothing_cross_entropy", "smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationEfficientFormerModelConfig(ModelConfig):
    task: str = "segmentation"
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
    checkpoint: Optional[Union[Path, str]] = "./weights/efficientformer/efficientformer_l1_1000d.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: EfficientFormerArchitectureConfig(
        head={"name": "faster_rcnn"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "roi_head_loss", "weight": None},
        {"criterion": "rpn_loss", "weight": None},
    ])


@dataclass
class ClassificationMobileNetV3ModelConfig(ModelConfig):
    task: str = "classification"
    checkpoint: Optional[Union[Path, str]] = "./weights/mobilenetv3/mobilenet_v3_small.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: MobileNetV3ArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "label_smoothing_cross_entropy", "smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationMobileNetV3ModelConfig(ModelConfig):
    task: str = "segmentation"
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
    checkpoint: Optional[Union[Path, str]] = "./weights/mobilevit/mobilevit_s.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: MobileViTArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "label_smoothing_cross_entropy", "smoothing": 0.1, "weight": None}
    ])


@dataclass
class PIDNetModelConfig(ModelConfig):
    task: str = "classification"
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
    checkpoint: Optional[Union[Path, str]] = "./weights/resnet/resnet50.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: ResNetArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "label_smoothing_cross_entropy", "smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationResNetModelConfig(ModelConfig):
    task: str = "segmentation"
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
    checkpoint: Optional[Union[Path, str]] = "./weights/segformer/segformer.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: SegFormerArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "label_smoothing_cross_entropy", "smoothing": 0.1, "weight": None}
    ])


@dataclass
class SegmentationSegFormerModelConfig(ModelConfig):
    task: str = "segmentation"
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
    checkpoint: Optional[Union[Path, str]] = "./weights/vit/vit-tiny.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: ViTArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "label_smoothing_cross_entropy", "smoothing": 0.1, "weight": None}
    ])

