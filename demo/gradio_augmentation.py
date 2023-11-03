import argparse
import os
from pathlib import Path

import gradio as gr
import netspresso_trainer
import numpy as np
import PIL.Image as Image
import torch
from netspresso_trainer.dataloaders import CREATE_TRANSFORM
from netspresso_trainer.loggers import VISUALIZER
from netspresso_trainer.models import SUPPORTING_MODEL_LIST, SUPPORTING_TASK_LIST
from omegaconf import OmegaConf

__version__ = netspresso_trainer.__version__

PATH_DOCS = os.getenv("PATH_DOCS", default="docs/description_augmentation.md")
PATH_EXAMPLE_CONFIG = os.getenv("PATH_EXAMPLE_CONFIG", default="config/augmentation_template_common.yaml")
PATH_EXAMPLE_IMAGE = os.getenv("PATH_EXAMPLE_IMAGE", default="assets/kyunghwan_cat.jpg")

NUM_AUGMENTATION_SAMPLES = 5


def summary_transform(phase, task, model_name, yaml_str):
    try:
        conf = OmegaConf.create(yaml_str)
        is_training = (phase == 'train')
        transform = CREATE_TRANSFORM(model_name, is_training=is_training)
        transform_composed = transform(conf.augmentation)
        return str(transform_composed)
    except Exception as e:
        raise gr.Error(str(e))


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
        raise gr.Error(str(e))


def parse_args():

    parser = argparse.ArgumentParser(description="Augmentation simulator for NetsPresso Trainer")

    # -------- User arguments ----------------------------------------

    parser.add_argument(
        '--docs', type=Path, default=PATH_DOCS,
        help="Docs string file")

    parser.add_argument(
        '--config', type=Path, default=PATH_EXAMPLE_CONFIG,
        help="Config for data augmentation")

    parser.add_argument(
        '--image', type=Path, default=PATH_EXAMPLE_IMAGE,
        help="Path for an example image")

    parser.add_argument(
        '--local', action='store_true',
        help="Whether to run in local environment or not")

    parser.add_argument(
        '--port', type=int, default=50003,
        help="Service port (only applicable when running on local server)")

    args, _ = parser.parse_known_args()

    return args


def launch_gradio(args):
    with gr.Blocks(theme='nota-ai/theme', title="Data Augmentation Simulator with NetsPresso Trainer") as demo:
        gr.Markdown(args.docs.read_text())
        gr.Markdown(f"<center>Package version: <code>netspresso-trainer=={__version__}</code></center>")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                task_choices = gr.Radio(label="Task: ", value='classification', choices=SUPPORTING_TASK_LIST)
            with gr.Column(scale=1):
                phase_choices = gr.Radio(label="Phase: ", value='train', choices=['train', 'valid'])
        model_choices = gr.Radio(label="Model: ", value='resnet50', choices=SUPPORTING_MODEL_LIST)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                config_input = gr.Code(label="Augmentation configuration", value=args.config.read_text(), language='yaml', lines=30)
            with gr.Column(scale=2):
                transform_repr_output = gr.Code(label="Data transform", lines=10)
                transform_button = gr.Button(value="Compose transform", variant='primary')
                test_image = gr.Image(value=str(args.image), label="Test image", type='pil')
        run_button = gr.Button(value="Get augmented samples", variant='primary')
        augmented_images = gr.Gallery(label="Results", columns=5)

        transform_compose_inputs = [phase_choices, task_choices, model_choices, config_input]
        run_inputs = transform_compose_inputs + [test_image]
        transform_button.click(summary_transform, inputs=transform_compose_inputs, outputs=transform_repr_output)
        run_button.click(get_augmented_images, inputs=run_inputs, outputs=augmented_images)

    if args.local:
        demo.launch(server_name="0.0.0.0", server_port=args.port)
    else:
        demo.launch()


if __name__ == "__main__":
    args = parse_args()
    launch_gradio(args)
