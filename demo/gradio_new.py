import argparse
import os
from pathlib import Path
from typing import List, Tuple, Union


import netspresso_trainer
import torch
import gradio as gr
import numpy as np
import pandas as pd
import PIL.Image as Image
from omegaconf import OmegaConf

from netspresso_trainer.dataloaders import CREATE_TRANSFORM
from netspresso_trainer.loggers import VISUALIZER, START_EPOCH_ZERO_OR_ONE
from netspresso_trainer.models import SUPPORTING_MODEL_LIST, SUPPORTING_TASK_LIST
from netspresso_trainer.optimizers import build_optimizer
from netspresso_trainer.schedulers import build_scheduler

__version__ = netspresso_trainer.__version__

PATH_AUG_DOCS = os.getenv(
    "PATH_AUG_DOCS", default="docs/description_augmentation.md")
PATH_AUG_EXAMPLE_CONFIG = os.getenv(
    "PATH_AUG_EXAMPLE_CONFIG", default="config/augmentation_template_common.yaml")
PATH_SCHEDULER_DOCS = os.getenv(
    "PATH_SCHEDULER_DOCS", default="docs/description_lr_scheduler.md")
PATH_SCHEDULER_EXAMPLE_CONFIG = os.getenv(
    "PATH_SCHEDULER_EXAMPLE_CONFIG", default="config/training_template_common.yaml")
PATH_EXAMPLE_IMAGE = os.getenv(
    "PATH_EXAMPLE_IMAGE", default="assets/kyunghwan_cat.jpg")

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


def get_lr_list(optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, total_epochs: int) -> List[float]:
    lr_list = []
    for _epoch in range(total_epochs):
        lr = scheduler.get_last_lr()[0]
        lr_list.append(lr)
        optimizer.step()
        scheduler.step()
    return lr_list


def get_lr_dataframe_from_config(yaml_str: str):
    try:
        conf = OmegaConf.create(yaml_str)
        model_mock = torch.nn.Linear(1, 1)
        optimizer = build_optimizer(model_mock,
                                    opt=conf.training.opt,
                                    lr=conf.training.lr,
                                    wd=conf.training.weight_decay,
                                    momentum=conf.training.momentum)
        scheduler, total_epochs = build_scheduler(optimizer, conf.training)
        lr_list = get_lr_list(optimizer, scheduler, total_epochs)

        df = pd.DataFrame({
            "lr": lr_list,
            "epochs": list(range(START_EPOCH_ZERO_OR_ONE, total_epochs + START_EPOCH_ZERO_OR_ONE))
        })
        return gr.LinePlot.update(df, x="epochs", y="lr", width=600, height=300, tooltip=["epochs", "lr"])
    except Exception as e:
        raise gr.Error(str(e))


def parse_args():

    parser = argparse.ArgumentParser(
        description="GUI for NetsPresso Trainer")

    # -------- User arguments ----------------------------------------

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
    with gr.Blocks(theme='nota-ai/theme', title="NetsPresso Trainer") as demo:
        gr.Markdown("\n\n# <center>Welcome to NetsPresso Trainer!</center>\n\n")
        with gr.Tab("Augmentation"):
            gr.Markdown(Path(PATH_AUG_DOCS).read_text())
            gr.Markdown(
                f"<center>Package version: <code>netspresso-trainer=={__version__}</code></center>")
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
                with gr.Column(scale=1):
                    config_input = gr.Code(label="Augmentation configuration",
                                           value=Path(PATH_AUG_EXAMPLE_CONFIG).read_text(), language='yaml', lines=30)
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

            transform_compose_inputs = [phase_choices,
                                        task_choices, model_choices, config_input]
            run_inputs = transform_compose_inputs + [test_image]
            transform_button.click(
                summary_transform, inputs=transform_compose_inputs, outputs=transform_repr_output)
            run_button.click(get_augmented_images,
                             inputs=run_inputs, outputs=augmented_images)

        with gr.Tab("Scheduler"):
            gr.Markdown(Path(PATH_SCHEDULER_DOCS).read_text())
            gr.Markdown(
                f"<center>Package version: <code>netspresso-trainer-{__version__}</code></center>")
            with gr.Row(equal_height=True):
                with gr.Column():
                    config_input = gr.Code(
                        label="Training configuration", value=Path(PATH_SCHEDULER_EXAMPLE_CONFIG).read_text(), language='yaml', lines=30)
                    run_button = gr.Button(value="Run", variant='primary')
                with gr.Column():
                    plot_output = gr.LinePlot()

            run_button.click(get_lr_dataframe_from_config,
                             inputs=config_input, outputs=plot_output)

    if args.local:
        demo.launch(server_name="0.0.0.0", server_port=args.port)
    else:
        demo.launch()


if __name__ == "__main__":
    args = parse_args()
    launch_gradio(args)
