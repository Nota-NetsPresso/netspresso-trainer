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
    example_model = ClassificationResNetModelConfig()
    
    # ### If you try to train torch.fx model from PyNetsPresso, use this block instead
    # example_model = ClassificationResNetModelConfig(checkpoint=None)
    # example_model.checkpoint.fx_model_path = "classification_resnet50_best_fx.pt"
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
        
    logging_dir = train_with_config(
        # gpus="0",
        config=cfg,
        log_level='INFO'
    )
    return logging_dir

def train_with_inline_yaml():
    from netspresso_trainer import train_with_yaml
    logging_dir = train_with_yaml(
        # gpus="0,1",
        data="config/data/beans.yaml",
        augmentation="config/augmentation/classification.yaml",
        model="config/model/resnet/resnet50-classification.yaml",
        training="config/training/classification.yaml",
        logging="config/logging.yaml",
        environment="config/environment.yaml",
        log_level='INFO'
    )
    return logging_dir


if __name__ == '__main__':
    # logging_dir = train_cli()

    # With inline yaml
    # logging_dir = train_with_inline_yaml()
    
    # With inline pythonic config
    logging_dir = train_with_inline_cfg()
    
    print(f"Training results are saved at: {logging_dir}")