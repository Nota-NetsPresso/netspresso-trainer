import os
from pathlib import Path

PATH_CONFIG_ROOT = os.getenv("PATH_CONFIG_ROOT", default="config/")
DEFAULT_MODEL_DICT = {
    "classification": "resnet50",
    "segmentation": "resnet50"
}
CONFIG_MODEL_DICT = {
    "classification": {
        "resnet50": "model/resnet/resnet50-classification.yaml",
        "efficientformer": "model/efficientformer/efficientformer-l1-classification.yaml",
        "mobilenetv3_small": "model/mobilenetv3/mobilenetv3-small-classification.yaml",
        "mobilevit": "model/mobilevit/mobilevit-s-classification.yaml",
        "segformer": "model/segformer/segformer-classification.yaml",
        "vit": "model/vit/vit-classification.yaml",
    },
    "segmentation": {
        "resnet50": "model/resnet/resnet50-segmentation.yaml",
        "efficientformer": "model/efficientformer/efficientformer-l1-segmentation.yaml",
        "mobilenetv3_small": "model/mobilenetv3/mobilenetv3-small-segmentation.yaml",
        "pidnet": "model/pidnet/pidnet-s-segmentation.yaml",
        "segformer": "model/segformer/segformer-segmentation.yaml",
    },
    "detection": {
        "efficientformer": "model/efficientformer/efficientformer-l1-detection.yaml",
        "yolox": "model/yolox/yolox-detection.yaml",
    }
}

CONFIG_AUGMENTATION_DICT = {
    "classification": "augmentation/classification.yaml",
    "segmentation": "augmentation/segmentation.yaml",
    "detection": "augmentation/detection.yaml",
}


def load_model_config(task, model) -> str:
    target_config_path = CONFIG_MODEL_DICT[task][model]
    return (Path(PATH_CONFIG_ROOT) / target_config_path).read_text()


def load_augmentation_config(task) -> str:
    target_config_path = CONFIG_AUGMENTATION_DICT[task]
    return (Path(PATH_CONFIG_ROOT) / target_config_path).read_text()
