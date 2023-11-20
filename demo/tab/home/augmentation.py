import argparse
import os
from pathlib import Path

import gradio as gr
from func.augmentation import get_augmented_images, summary_transform
from netspresso_trainer.models import SUPPORTING_MODEL_LIST

# TODO: directly import from netspresso_trainer.models
SUPPORTING_TASK_LIST = ['classification', 'segmentation']

PATH_AUG_DOCS = os.getenv(
    "PATH_AUG_DOCS", default="docs/description_augmentation.md")
PATH_AUG_EXAMPLE_CONFIG = os.getenv(
    "PATH_AUG_EXAMPLE_CONFIG", default="config/augmentation_template_common.yaml")


def tab_augmentation(args):
    gr.Markdown(Path(PATH_AUG_DOCS).read_text())
    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            task_choices = gr.Radio(
                label="Task: ", value='classification', choices=SUPPORTING_TASK_LIST)
        with gr.Column(scale=1):
            phase_choices = gr.Radio(
                label="Phase: ", value='train', choices=['train', 'valid'])
    model_choices = gr.Radio(
        label="Model: ", value='resnet50', choices=SUPPORTING_MODEL_LIST)
    with gr.Row(equal_height=True):
        with gr.Column(scale=1), gr.Group():
            config_input = gr.Code(label="Augmentation configuration",
                                   value=Path(PATH_AUG_EXAMPLE_CONFIG).read_text(),
                                   language='yaml', lines=30)
            with gr.Row():
                go_back_button = gr.Button(value="Back to Train", variant='secondary')
                config_copy_button = gr.Button(value="Copy to Train", variant='secondary')
        with gr.Column(scale=2):
            transform_repr_output = gr.Code(
                label="Data transform", lines=10)
            transform_button = gr.Button(
                value="Compose transform", variant='primary')
            test_image = gr.Image(
                value=str(args.image), label="Test image", type='pil')
    run_button = gr.Button(
        value="Get augmented samples", variant='primary')
    augmented_images = gr.Gallery(label="Results", columns=5)

    transform_compose_inputs = [phase_choices, task_choices, model_choices, config_input]
    run_inputs = transform_compose_inputs + [test_image]
    transform_button.click(
        summary_transform,
        inputs=transform_compose_inputs,
        outputs=transform_repr_output
    )
    run_button.click(
        get_augmented_images,
        inputs=run_inputs,
        outputs=augmented_images
    )

    return config_input, config_copy_button, go_back_button
