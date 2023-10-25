from omegaconf import OmegaConf
from copy import deepcopy

from netspresso_trainer.cfg import TrainerConfig, ClassificationAugmentationConfig, ColorJitter

if __name__ == '__main__':
    augmentation_config = ClassificationAugmentationConfig(color_jitter=ColorJitter(colorjitter_p=0.9))
    cfg = TrainerConfig(augmentation=augmentation_config)
    
    # Basic: convert dataclass into yaml config
    config: TrainerConfig = OmegaConf.structured(cfg)
    print(OmegaConf.to_yaml(config))
    
    # OK: update value from OmegaConf Config
    config_new: TrainerConfig = deepcopy(config)
    config_new.augmentation.color_jitter.hue = 0.5
    print(OmegaConf.to_yaml(config_new))
    
    # OK: update value of subclass in the main dataclass
    cfg_new: TrainerConfig = deepcopy(cfg)
    cfg_new.augmentation.color_jitter.saturation = 0.0
    print(OmegaConf.to_yaml(OmegaConf.structured(cfg_new)))
    