from netspresso_trainer import train_cli

def train_with_inline_cfg():
    from netspresso_trainer import TrainerConfig, train_with_config, export_config_as_yaml  
    from netspresso_trainer.cfg import ClassificationResNetModelConfig, ClassificationMixNetMediumModelConfig, ExampleBeansDataset
    
    """
    Declare dataset config (Use an example dataset provided from the module)
    """
    example_dataset = ExampleBeansDataset
    example_dataset.metadata.custom_cache_dir = "./data/huggingface"
    """
    Declare model config
    """
    example_model = ClassificationMixNetMediumModelConfig()
    
    # ### If you try to train torch.fx model from PyNetsPresso, use this block instead
    # example_model = ClassificationResNetModelConfig(checkpoint=None)
    # example_model.fx_model_checkpoint = "classification_resnet50_best_fx.pt"
    # ###
    
    
    """
    Declare trainer config
    """
    cfg = TrainerConfig(
        task='classification',
        auto=True,
        data=example_dataset,
        model=example_model
    )
    
    """
    Update major field values considering the spec of training machine
    """
    cfg.epochs = 3
    cfg.logging.csv = True
    cfg.environment.gpus = "1"
        
    train_with_config(
        # gpus="0",
        config=cfg,
        log_level='INFO'
    )

def train_with_inline_yaml():
    from netspresso_trainer import train_with_yaml
    train_with_yaml(
        data="config/data/beans.yaml",
        augmentation="config/augmentation/classification.yaml",
        model="config/model/resnet/resnet50-classification.yaml",
        training="config/training/classification.yaml",
        logging="config/logging.yaml",
        environment="config/environment.yaml",
        log_level='DEBUG'
    )


if __name__ == '__main__':
    # train_cli()

    # With inline yaml
    train_with_inline_yaml()
    
    # With inline pythonic config
    # train_with_inline_cfg()