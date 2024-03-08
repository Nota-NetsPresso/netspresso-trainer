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
class LocalPoseEstimationDatasetConfig(DatasetConfig):
    task: str = "pose_estimation"
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

ExampleWFLWDataset = LocalPoseEstimationDatasetConfig(
    name="wflw",
    path=PathConfig(
        root="../../data/WFLW",
        train=ImageLabelPathConfig(image="images/train", label="labels/train"),
        valid=ImageLabelPathConfig(image="images/val", label="labels/val"),
    ),
    id_mapping=[
        {'name': '0', 'skeleton': None, 'swap': '32'},
        {'name': '1', 'skeleton': None, 'swap': '31'},
        {'name': '2', 'skeleton': None, 'swap': '30'},
        {'name': '3', 'skeleton': None, 'swap': '29'},
        {'name': '4', 'skeleton': None, 'swap': '28'},
        {'name': '5', 'skeleton': None, 'swap': '27'},
        {'name': '6', 'skeleton': None, 'swap': '26'},
        {'name': '7', 'skeleton': None, 'swap': '25'},
        {'name': '8', 'skeleton': None, 'swap': '24'},
        {'name': '9', 'skeleton': None, 'swap': '23'},
        {'name': '10', 'skeleton': None, 'swap': '22'},
        {'name': '11', 'skeleton': None, 'swap': '21'},
        {'name': '12', 'skeleton': None, 'swap': '20'},
        {'name': '13', 'skeleton': None, 'swap': '19'},
        {'name': '14', 'skeleton': None, 'swap': '18'},
        {'name': '15', 'skeleton': None, 'swap': '17'},
        {'name': '16', 'skeleton': None, 'swap': '16'},
        {'name': '17', 'skeleton': None, 'swap': '15'},
        {'name': '18', 'skeleton': None, 'swap': '14'},
        {'name': '19', 'skeleton': None, 'swap': '13'},
        {'name': '20', 'skeleton': None, 'swap': '12'},
        {'name': '21', 'skeleton': None, 'swap': '11'},
        {'name': '22', 'skeleton': None, 'swap': '10'},
        {'name': '23', 'skeleton': None, 'swap': '9'},
        {'name': '24', 'skeleton': None, 'swap': '8'},
        {'name': '25', 'skeleton': None, 'swap': '7'},
        {'name': '26', 'skeleton': None, 'swap': '6'},
        {'name': '27', 'skeleton': None, 'swap': '5'},
        {'name': '28', 'skeleton': None, 'swap': '4'},
        {'name': '29', 'skeleton': None, 'swap': '3'},
        {'name': '30', 'skeleton': None, 'swap': '2'},
        {'name': '31', 'skeleton': None, 'swap': '1'},
        {'name': '32', 'skeleton': None, 'swap': '0'},
        {'name': '33', 'skeleton': None, 'swap': '46'},
        {'name': '34', 'skeleton': None, 'swap': '45'},
        {'name': '35', 'skeleton': None, 'swap': '44'},
        {'name': '36', 'skeleton': None, 'swap': '43'},
        {'name': '37', 'skeleton': None, 'swap': '42'},
        {'name': '38', 'skeleton': None, 'swap': '50'},
        {'name': '39', 'skeleton': None, 'swap': '49'},
        {'name': '40', 'skeleton': None, 'swap': '48'},
        {'name': '41', 'skeleton': None, 'swap': '47'},
        {'name': '42', 'skeleton': None, 'swap': '37'},
        {'name': '43', 'skeleton': None, 'swap': '36'},
        {'name': '44', 'skeleton': None, 'swap': '35'},
        {'name': '45', 'skeleton': None, 'swap': '34'},
        {'name': '46', 'skeleton': None, 'swap': '33'},
        {'name': '47', 'skeleton': None, 'swap': '41'},
        {'name': '48', 'skeleton': None, 'swap': '40'},
        {'name': '49', 'skeleton': None, 'swap': '39'},
        {'name': '50', 'skeleton': None, 'swap': '38'},
        {'name': '51', 'skeleton': None, 'swap': '51'},
        {'name': '52', 'skeleton': None, 'swap': '52'},
        {'name': '53', 'skeleton': None, 'swap': '53'},
        {'name': '54', 'skeleton': None, 'swap': '54'},
        {'name': '55', 'skeleton': None, 'swap': '59'},
        {'name': '56', 'skeleton': None, 'swap': '58'},
        {'name': '57', 'skeleton': None, 'swap': '57'},
        {'name': '58', 'skeleton': None, 'swap': '56'},
        {'name': '59', 'skeleton': None, 'swap': '55'},
        {'name': '60', 'skeleton': None, 'swap': '72'},
        {'name': '61', 'skeleton': None, 'swap': '71'},
        {'name': '62', 'skeleton': None, 'swap': '70'},
        {'name': '63', 'skeleton': None, 'swap': '69'},
        {'name': '64', 'skeleton': None, 'swap': '68'},
        {'name': '65', 'skeleton': None, 'swap': '75'},
        {'name': '66', 'skeleton': None, 'swap': '74'},
        {'name': '67', 'skeleton': None, 'swap': '73'},
        {'name': '68', 'skeleton': None, 'swap': '64'},
        {'name': '69', 'skeleton': None, 'swap': '63'},
        {'name': '70', 'skeleton': None, 'swap': '62'},
        {'name': '71', 'skeleton': None, 'swap': '61'},
        {'name': '72', 'skeleton': None, 'swap': '60'},
        {'name': '73', 'skeleton': None, 'swap': '67'},
        {'name': '74', 'skeleton': None, 'swap': '66'},
        {'name': '75', 'skeleton': None, 'swap': '65'},
        {'name': '76', 'skeleton': None, 'swap': '82'},
        {'name': '77', 'skeleton': None, 'swap': '81'},
        {'name': '78', 'skeleton': None, 'swap': '80'},
        {'name': '79', 'skeleton': None, 'swap': '79'},
        {'name': '80', 'skeleton': None, 'swap': '78'},
        {'name': '81', 'skeleton': None, 'swap': '77'},
        {'name': '82', 'skeleton': None, 'swap': '76'},
        {'name': '83', 'skeleton': None, 'swap': '87'},
        {'name': '84', 'skeleton': None, 'swap': '86'},
        {'name': '85', 'skeleton': None, 'swap': '85'},
        {'name': '86', 'skeleton': None, 'swap': '84'},
        {'name': '87', 'skeleton': None, 'swap': '83'},
        {'name': '88', 'skeleton': None, 'swap': '92'},
        {'name': '89', 'skeleton': None, 'swap': '91'},
        {'name': '90', 'skeleton': None, 'swap': '90'},
        {'name': '91', 'skeleton': None, 'swap': '89'},
        {'name': '92', 'skeleton': None, 'swap': '88'},
        {'name': '93', 'skeleton': None, 'swap': '95'},
        {'name': '94', 'skeleton': None, 'swap': '94'},
        {'name': '95', 'skeleton': None, 'swap': '93'},
        {'name': '96', 'skeleton': None, 'swap': '97'},
        {'name': '97', 'skeleton': None, 'swap': '96'},
    ],
)
