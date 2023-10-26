from netspresso_trainer import train_with_config, TrainerConfig


if __name__ == '__main__':
    from netspresso_trainer.cfg import (
        ClassificationAugmentationConfig,
        ClassificationResNetModelConfig,
        ExampleBeansDataset,
        ColorJitter
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
    
    train_with_config(cfg, log_level='INFO')