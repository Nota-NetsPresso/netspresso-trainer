import gradio as gr
from netspresso_trainer.dataloaders import CREATE_TRANSFORM
from omegaconf import OmegaConf

NUM_AUGMENTATION_SAMPLES = 8


def summary_transform(phase, task, model_name, yaml_str):
    try:
        conf = OmegaConf.create(yaml_str)
        is_training = (phase == 'train')
        transform = CREATE_TRANSFORM(model_name, is_training=is_training)
        transform_composed = transform(conf.augmentation)
        return str(transform_composed)
    except Exception as e:
        raise gr.Error(str(e)) from e


def get_augmented_images(phase, task, model_name, yaml_str, test_image,
                         num_samples=NUM_AUGMENTATION_SAMPLES):
    try:
        conf = OmegaConf.create(yaml_str)
        is_training = (phase == 'train')
        transform = CREATE_TRANSFORM(model_name, is_training=is_training)
        transform_composed = transform(conf.augmentation)

        transformed_images = [transform_composed(test_image,
                                                 visualize_for_debug=True)['image']
                              for _ in range(num_samples)]
        return transformed_images
    except Exception as e:
        raise gr.Error(str(e)) from e
