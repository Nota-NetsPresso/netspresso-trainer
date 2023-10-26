from copy import deepcopy

import omegaconf
from netspresso_trainer.cfg import TrainerConfig
from omegaconf import OmegaConf
from pytest import raises

if __name__ == "__main__":
    
    from netspresso_trainer.cfg import (
        ClassificationAugmentationConfig,
        ClassificationResNetModelConfig,
        ColorJitter,
        ExampleBeansDataset,
    )
    
    augmentation_config = ClassificationAugmentationConfig(color_jitter=ColorJitter(colorjitter_p=0.9))
    example_dataset = ExampleBeansDataset
    example_model = ClassificationResNetModelConfig()
    cfg = TrainerConfig(
        task='classification',
        auto=True,
        data=ExampleBeansDataset,
        model=example_model,
        augmentation=augmentation_config
    )
    
    
    # Basic: convert dataclass into yaml config
    config: TrainerConfig = OmegaConf.structured(cfg)
    print(OmegaConf.to_yaml(config))
    
    # OK: update value of subclass in the main dataclass
    cfg_new: TrainerConfig = deepcopy(cfg)
    cfg_new.augmentation.color_jitter.saturation = 0.0
    # print(OmegaConf.to_yaml(OmegaConf.structured(cfg_new)))
    
    # OK: update value from OmegaConf Config
    config_new: TrainerConfig = deepcopy(config)
    config_new.augmentation.color_jitter.hue = 0.5
    # print(OmegaConf.to_yaml(config_new))


    # OK: some necessary fields can be updated with shortcut before converting to DictConfig
    cfg_shortcut: TrainerConfig = deepcopy(cfg)
    cfg_shortcut.epochs = 1010
    cfg_shortcut.batch_size = 128
    # print(OmegaConf.to_yaml(OmegaConf.structured(cfg_shortcut)))
    
    # FAIL: but not with DictConfig itself from OmegaConf
    with raises(omegaconf.errors.ConfigAttributeError):
        config_shortcut: TrainerConfig = deepcopy(config)
        config_shortcut.batch_size = 256
        # print(OmegaConf.to_yaml(config_shortcut))