from netspresso_trainer import TrainerConfig, train, export_config_as_yaml

if __name__ == '__main__':
    from netspresso_trainer.cfg import ClassificationResNetModelConfig, ExampleBeansDataset
    
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
    # example_model.fx_model_checkpoint = "classification_resnet50_best_fx.pt"
    # ###
    
    
    """
    Declare trainer config
    """
    cfg = TrainerConfig(
        task='classification',
        auto=True,
        data=ExampleBeansDataset,
        model=example_model
    )
    
    """
    Update major field values considering the spec of training machine
    """
    cfg.batch_size = 64
    cfg.epochs = 5
    
    print(export_config_as_yaml(cfg))
    
    train(cfg, log_level='INFO')