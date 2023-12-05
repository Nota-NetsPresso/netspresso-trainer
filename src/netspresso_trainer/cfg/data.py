from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from omegaconf import MISSING, MissingMandatoryValue

__all__ = [
    "DatasetConfig",
    "LocalClassificationDatasetConfig",
    "LocalSegmentationDatasetConfig",
    "LocalDetectionDatasetConfig",
    "HuggingFaceClassificationDatasetConfig",
    "HuggingFaceSegmentationDatasetConfig",
    "ExampleBeansDataset",
    "ExampleChessDataset",
    "ExampleCocoyoloDataset",
    "ExampleSidewalkDataset",
    "ExampleXrayDataset",
    "ExampleSidewalkDataset",
    "ExampleSkincancerDataset",
    "ExampleTrafficsignDataset",
    "ExampleVoc12Dataset",
    "ExampleVoc12CustomDataset",
    "ExampleWikiartDataset",
]


@dataclass
class DatasetConfig:
    name: str = MISSING
    task: str = MISSING
    format: str = MISSING  # Literal['huggingface', 'local']


@dataclass
class ImageLabelPathConfig:
    image: Optional[Union[Path, str]] = None
    label: Optional[Union[Path, str]] = None


@dataclass
class PathPatternConfig:
    image: Optional[str] = None
    label: Optional[str] = None


@dataclass
class PathConfig:
    root: Union[Path, str] = MISSING
    train: ImageLabelPathConfig = field(default_factory=lambda: ImageLabelPathConfig(image=MISSING))
    valid: ImageLabelPathConfig = field(default_factory=lambda: ImageLabelPathConfig())
    test: ImageLabelPathConfig = field(default_factory=lambda: ImageLabelPathConfig())
    pattern: PathPatternConfig = field(default_factory=lambda: PathPatternConfig())


@dataclass
class HuggingFaceConfig:
    custom_cache_dir: Optional[Union[Path, str]] = None # If None, it follows HF datasets default (.cache/huggingface/datasets)
    repo: str = MISSING
    subset: Optional[str] = None
    features: Dict[str, str] = field(default_factory=lambda: {
        "image": "image", "label": "labels"
    })


@dataclass
class LocalClassificationDatasetConfig(DatasetConfig):
    task: str = "classification"
    format: str = "local"
    path: PathConfig = field(default_factory=lambda: PathConfig())
    id_mapping: Optional[List[str]] = None


@dataclass
class LocalSegmentationDatasetConfig(DatasetConfig):
    task: str = "segmentation"
    format: str = "local"
    path: PathConfig = field(default_factory=lambda: PathConfig())
    label_image_mode: str = "L"
    id_mapping: Any = None
    pallete: Optional[List[List[int]]] = None


@dataclass
class LocalDetectionDatasetConfig(DatasetConfig):
    task: str = "detection"
    format: str = "local"
    path: PathConfig = field(default_factory=lambda: PathConfig())
    id_mapping: Any = None
    pallete: Optional[List[List[int]]] = None


@dataclass
class HuggingFaceClassificationDatasetConfig(DatasetConfig):
    task: str = "classification"
    format: str = "huggingface"
    metadata: HuggingFaceConfig = field(default_factory=lambda: HuggingFaceConfig(
        features={"image": "image", "label": "labels"}
    ))
    id_mapping: Optional[List[str]] = None


@dataclass
class HuggingFaceSegmentationDatasetConfig(DatasetConfig):
    task: str = "segmentation"
    format: str = "huggingface"
    metadata: HuggingFaceConfig = field(default_factory=lambda: HuggingFaceConfig(
        features={"image": "pixel_values", "label": "label"}
    ))
    label_image_mode: str = "L"
    id_mapping: Any = None
    pallete: Optional[List[List[int]]] = None


ExampleBeansDataset = HuggingFaceClassificationDatasetConfig(
    name="beans",
    metadata=HuggingFaceConfig(
        custom_cache_dir=None,
        repo="beans",
        features={"image": "image", "label": "labels"}
    )
)

ExampleChessDataset = LocalClassificationDatasetConfig(
    name="chess",
    path=PathConfig(
        root="/DATA/classification-example",
        train=ImageLabelPathConfig(image="train"),
        valid=ImageLabelPathConfig(image="val"),
    )
)

ExampleXrayDataset = HuggingFaceClassificationDatasetConfig(
    name="chest_xray_classification",
    metadata=HuggingFaceConfig(
        custom_cache_dir=None,
        repo="keremberke/chest-xray-classification",
        subset="full",
        features={"image": "image", "label": "labels"}
    )
)

ExampleCocoyoloDataset = LocalDetectionDatasetConfig(
    name="coco_for_yolo_model",
    path=PathConfig(
        root="/DATA/coco",
        train=ImageLabelPathConfig(image="images/train2017", label="labels/train2017"),
        valid=ImageLabelPathConfig(image="images/train2017", label="labels/train2017"),
        pattern=PathPatternConfig(image="([0-9]{12})\\.jpg", label="([0-9]{12})\\.txt"),
    ),
    id_mapping=[
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
)

ExampleSidewalkDataset = HuggingFaceSegmentationDatasetConfig(
    name="sidewalk_semantic",
    metadata=HuggingFaceConfig(
        custom_cache_dir=None,
        repo="segments/sidewalk-semantic",
        features={"image": "pixel_values", "label": "label"}
    ),
    label_image_mode="L",
    id_mapping=[
        'unlabeled', 'flat-road', 'flat-sidewalk', 'flat-crosswalk', 'flat-cyclinglane', 'flat-parkingdriveway',
        'flat-railtrack', 'flat-curb', 'human-person', 'human-rider', 'vehicle-car', 'vehicle-truck', 'vehicle-bus',
        'vehicle-tramtrain', 'vehicle-motorcycle', 'vehicle-bicycle', 'vehicle-caravan', 'vehicle-cartrailer',
        'construction-building', 'construction-door', 'construction-wall', 'construction-fenceguardrail',
        'construction-bridge', 'construction-tunnel', 'construction-stairs', 'object-pole', 'object-trafficsign',
        'object-trafficlight', 'nature-vegetation', 'nature-terrain', 'sky', 'void-ground', 'void-dynamic',
        'void-static', 'void-unclear'
    ]
)

ExampleSkincancerDataset = HuggingFaceClassificationDatasetConfig(
    name="skin_cancer",
    metadata=HuggingFaceConfig(
        custom_cache_dir=None,
        repo="marmal88/skin_cancer",
        features={"image": "image", "label": "dx"}
    )
)

ExampleTrafficsignDataset = LocalDetectionDatasetConfig(
    name="traffic_sign_yolo",
    path=PathConfig(
        root="../../data/traffic-sign",
        train=ImageLabelPathConfig(image="images/train", label="labels/train"),
        valid=ImageLabelPathConfig(image="images/val", label="labels/val"),
    ),
    id_mapping=['prohibitory', 'danger', 'mandatory', 'other']  # class names
)

ExampleVoc12Dataset = LocalSegmentationDatasetConfig(
    name="voc2012",
    path=PathConfig(
        root="/DATA/VOC12Dataset",
        train=ImageLabelPathConfig(image="image/train", label="mask/train"),
        valid=ImageLabelPathConfig(image="image/val", label="mask/val"),
    ),
    label_image_mode="RGB",
    id_mapping={
        "(0, 0, 0)": "background",
        "(128, 0, 0)": "aeroplane",
        "(0, 128, 0)": "bicycle",
        "(128, 128, 0)": "bird",
        "(0, 0, 128)": "boat",
        "(128, 0, 128)": "bottle",
        "(0, 128, 128)": "bus",
        "(128, 128, 128)": "car",
        "(64, 0, 0)": "cat",
        "(192, 0, 0)": "chair",
        "(64, 128, 0)": "cow",
        "(192, 128, 0)": "diningtable",
        "(64, 0, 128)": "dog",
        "(192, 0, 128)": "horse",
        "(64, 128, 128)": "motorbike",
        "(192, 128, 128)": "person",
        "(0, 64, 0)": "pottedplant",
        "(128, 64, 0)": "sheep",
        "(0, 192, 0)": "sofa",
        "(128, 192, 0)": "train",
        "(0, 64, 128)": "tvmonitor",
        "(128, 64, 128)": "void"
    }
)

ExampleVoc12CustomDataset = LocalSegmentationDatasetConfig(
    name="voc2012",
    path=PathConfig(
        root="../../data/VOC12Dataset",
        train=ImageLabelPathConfig(image="image/train", label="mask/train"),
        valid=ImageLabelPathConfig(image="image/val", label="mask/val"),
    ),
    label_image_mode="L",
    id_mapping=[
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ],
    pallete=[
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
        [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
        [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
        [128, 192, 0], [0, 64, 128]
    ]
)

ExampleWikiartDataset = HuggingFaceClassificationDatasetConfig(
    name="wikiart_artist",
    metadata=HuggingFaceConfig(
        custom_cache_dir=None,
        repo="huggan/wikiart",
        subset="full",
        features={"image": "image", "label": "artist"}
    )
)
