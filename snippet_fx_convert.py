from utils.fx import convert_graphmodule

if __name__ == '__main__':
    from models.builder import AssembleModel
    from omegaconf import OmegaConf

    yaml_path = "config/resnet.yaml"
    config = OmegaConf.load(yaml_path)
    model = AssembleModel(config, num_classes=20)
    fx_model = convert_graphmodule(model)
    print(fx_model)
