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


def summary_transform(yaml_str):
    pass


def get_augmented_images(yaml_str, sample_image):
    pass


def launch_gradio(args):
    with gr.Blocks(theme='nota-ai/theme', title="Data Augmentation Simulator with NetsPresso Trainer") as demo:
        gr.Markdown((CURRENT_DIR / "docs" / "description_augmentation.md").read_text())
        gr.Markdown(f"<center>Package version: <code>netspresso-trainer-{__version__}</code></center>")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                with gr.Row(equal_height=True):
                    example_training_config_path = CURRENT_DIR.parent / "config" / "augmentation" / "template" / "common.yaml"
                    config_input = gr.Code(label="Augmentation configuration", value=example_training_config_path.read_text(), language='yaml', lines=30)
                    transform_repr_output = gr.Code(label="Data transform", lines=30)
                with gr.Row():
                    transform_button = gr.Button(value="Compose transform", variant='primary')
            with gr.Column(scale=1):
                test_image = gr.Image(label="Test image")
                phase_choices = gr.Radio(label="Phase: ", value='train', choices=['train', 'validation'])
        with gr.Row(equal_height=True):
            task_choices = gr.Radio(label="Task: ", value='classification', choices=SUPPORTING_TASK_LIST)
            model_choices = gr.Radio(label="Model: ", value='resnet50', choices=SUPPORTING_MODEL_LIST)
        with gr.Row():
            run_button = gr.Button(value="Get augmented samples", variant='primary')
        with gr.Row():
            augmented_images = gr.Gallery(label="Results")

        transform_button.click(summary_transform, inputs=config_input, outputs=transform_repr_output)
        run_inputs = [config_input, test_image, phase_choices, task_choices, model_choices]
        run_button.click(get_augmented_images, inputs=run_inputs, outputs=augmented_images)

    demo.launch(server_name="0.0.0.0", server_port=50003)


if __name__ == "__main__":
    launch_gradio(args=None)
