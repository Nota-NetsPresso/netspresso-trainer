import argparse
import os
from pathlib import Path

import gradio as gr
from func.augmentation import get_augmented_images, summary_transform

PATH_AUG_DOCS = os.getenv(
    "PATH_AUG_DOCS", default="docs/description_augmentation.md")
PATH_CONFIG_ROOT = os.getenv("PATH_CONFIG_ROOT", default="config/")
PATH_AUG_EXAMPLE_CONFIG = Path(PATH_CONFIG_ROOT) / "augmentation/classification.yaml"


def tab_augmentation(args, task_choices, model_choices):
    gr.Markdown(Path(PATH_AUG_DOCS).read_text())
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
                config_input = gr.Code(label="Augmentation configuration",
                                       value=Path(PATH_AUG_EXAMPLE_CONFIG).read_text(),
                                       language='yaml', lines=15)
                with gr.Row():
                    go_back_button = gr.Button(value="Back to Train", variant='secondary')
                    config_copy_button = gr.Button(value="Copy to Train", variant='secondary')

            test_image = gr.Image(
                value=str(args.image), label="Test image", type='pil'
            )
            phase_choices = gr.Radio(
                label="Phase: ", value='train', choices=['train', 'valid']
            )
            run_button = gr.Button(
                value="Get augmented samples", variant='primary'
            )
        with gr.Column(scale=2):
            transform_repr_output = gr.Code(
                label="Data transform", lines=10
            )

            augmented_images = gr.Gallery(label="Results", columns=4)

    transform_compose_inputs = [phase_choices, task_choices, model_choices, config_input]
    run_inputs = transform_compose_inputs + [test_image]
    run_button.click(
        summary_transform,
        inputs=transform_compose_inputs,
        outputs=transform_repr_output
    ).success(
        get_augmented_images,
        inputs=run_inputs,
        outputs=augmented_images
    )

    return config_input, config_copy_button, go_back_button
