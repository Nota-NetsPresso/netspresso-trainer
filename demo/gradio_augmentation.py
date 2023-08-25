from typing import Tuple, List, Union
from pathlib import Path

import gradio as gr
import numpy as np
import PIL.Image as Image
from omegaconf import OmegaConf
import torch

import netspresso_trainer
from netspresso_trainer.models import SUPPORTING_MODEL_LIST, SUPPORTING_TASK_LIST
from netspresso_trainer.dataloaders import CREATE_TRANSFORM
from netspresso_trainer.loggers import VISUALIZER

__version__ = netspresso_trainer.__version__

CURRENT_DIR = Path(__file__).resolve().parent
NUM_AUGMENTATION_SAMPLES = 5


def summary_transform(phase, task, model_name, yaml_str):
    conf = OmegaConf.create(yaml_str)
    is_training = (phase == 'train')
    transform = CREATE_TRANSFORM[task](model_name, is_training=is_training)
    transform_composed = transform(conf.augmentation)
    return str(transform_composed)


def get_augmented_images(phase, task, model_name, yaml_str, test_image,
                         num_samples=NUM_AUGMENTATION_SAMPLES):
    conf = OmegaConf.create(yaml_str)
    is_training = (phase == 'train')
    transform = CREATE_TRANSFORM[task](model_name, is_training=is_training)
    transform_composed = transform(conf.augmentation)

    transformed_images = [transform_composed(test_image,
                                             visualize_for_debug=True)['image']
                          for _ in range(num_samples)]
    return transformed_images


def launch_gradio(args):
    with gr.Blocks(theme='nota-ai/theme', title="Data Augmentation Simulator with NetsPresso Trainer") as demo:
        gr.Markdown((CURRENT_DIR / "docs" / "description_augmentation.md").read_text())
        gr.Markdown(f"<center>Package version: <code>netspresso-trainer-{__version__}</code></center>")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                task_choices = gr.Radio(label="Task: ", value='classification', choices=SUPPORTING_TASK_LIST)
            with gr.Column(scale=1):
                phase_choices = gr.Radio(label="Phase: ", value='train', choices=['train', 'valid'])
        model_choices = gr.Radio(label="Model: ", value='resnet50', choices=SUPPORTING_MODEL_LIST)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                example_training_config_path = CURRENT_DIR.parent / "config" / "augmentation" / "template" / "common.yaml"
                config_input = gr.Code(label="Augmentation configuration", value=example_training_config_path.read_text(), language='yaml', lines=30)
            with gr.Column(scale=2):
                transform_repr_output = gr.Code(label="Data transform", lines=10)
                transform_button = gr.Button(value="Compose transform", variant='primary')
                test_image = gr.Image(value=str(CURRENT_DIR.parent / "assets" / "kyunghwan_cat.jpg"), label="Test image", type='pil')
        run_button = gr.Button(value="Get augmented samples", variant='primary')
        augmented_images = gr.Gallery(label="Results", columns=5)

        transform_compose_inputs = [phase_choices, task_choices, model_choices, config_input]
        run_inputs = transform_compose_inputs + [test_image]
        transform_button.click(summary_transform, inputs=transform_compose_inputs, outputs=transform_repr_output)
        run_button.click(get_augmented_images, inputs=run_inputs, outputs=augmented_images)

    demo.launch(server_name="0.0.0.0", server_port=50003)


if __name__ == "__main__":
    launch_gradio(args=None)
