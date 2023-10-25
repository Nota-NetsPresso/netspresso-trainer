from pathlib import Path
from typing import Optional, Union, Dict, List, Any
from dataclasses import dataclass, field

from omegaconf import MISSING, MissingMandatoryValue

__all__ = ["ModelConfig", "ClassificationSegFormerModelConfig", "SegmentationSegFormerModelConfig"]

@dataclass
class ArchitectureConfig:
    full: Optional[Dict[str, Any]] = None
    backbone: Optional[Dict[str, Any]] = None
    head: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        assert bool(self.full) != bool(self.backbone), f"Only one of full or backbone should be given."
    
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
class ClassificationSegFormerModelConfig(ModelConfig):
    task: str = "classification"
    checkpoint: str = "./weights/segformer/segformer.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: SegFormerArchitectureConfig(
        head={"name": "fc"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "label_smoothing_cross_entropy", "smoothing": 0.1, "weight": None}
    ])



@dataclass
class SegmentationSegFormerModelConfig(ModelConfig):
    task: str = "segmentation"
    checkpoint: str = "./weights/segformer/segformer.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: SegFormerArchitectureConfig(
        head={"name": "all_mlp_decoder"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "cross_entropy", "ignore_index": 255, "weight": None}
    ])
