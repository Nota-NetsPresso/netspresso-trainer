import argparse
import os
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, TypedDict

import netspresso
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

from netspresso.client import SessionClient
from netspresso.compressor import ModelCompressor, Task, Framework

__version__netspresso_trainer = netspresso_trainer.__version__
__version__netspresso = netspresso.__version__


# TODO: directly import from netspresso_trainer.models
SUPPORTING_TASK_LIST = ['classification', 'segmentation']

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


def get_lr_list(
        optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, total_epochs: int) -> List[
        float]:
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


class InputShapes(TypedDict):
    batch: int
    channel: int
    dimension: List[int]  # Height, Width


class NetsPressoSession:
    task_dict = {
        'classification': Task.IMAGE_CLASSIFICATION,
        'segmentation': Task.SEMANTIC_SEGMENTATION
    }

    def __init__(self) -> None:
        self.compressor = None
        self._is_verified = False

    @property
    def is_verified(self) -> bool:
        return self._is_verified

    def login(self, email: str, password: str) -> bool:
        try:
            session = SessionClient(email=email, password=password)
            self.compressor = ModelCompressor(user_session=session)
            self._is_verified = True
        except Exception as e:
            self._is_verified = False
            raise e
        finally:
            return self._is_verified

    def compress(self, model_name: str, task: str, model_path: Union[Path, str],
                 batch_size: int, channels: int, height: int, width: int,
                 compression_ratio: float,
                 compressed_model_path: Optional[Union[Path, str]]) -> Path:

        if not self._is_verified:
            raise gr.Error(f"Please log in first at the console on the left side.")

        if self.compressor is None:
            self._is_verified = False
            raise gr.Error(f"The session is expired! Please log in again.")

        if task not in self.task_dict:
            raise gr.Error(f"Selected task is not supported in web UI version.")

        model = self.compressor.upload_model(
            model_name=model_name,
            task=self.task_dict[task],
            # file_path: e.g. ./model.pt
            file_path=str(model_path),
            # input_shapes: e.g. [{"batch": 1, "channel": 3, "dimension": [32, 32]}]
            input_shapes=[InputShapes(batch=batch_size, channel=channels, dimension=[height, width])],
            framework=Framework.PYTORCH
        )

        _ = self.compressor.automatic_compression(
            model_id=model.model_id,
            model_name=model_name,
            # output_path: e.g. ./compressed_model.pt
            output_path=str(compressed_model_path),
            compression_ratio=compression_ratio,
        )

        return Path(compressed_model_path)


def login_with_session(session: NetsPressoSession, email: str, password: str) -> NetsPressoSession:
    try:
        success = session.login(email, password)
        if success:
            gr.Info("Login success!")
            return session
    except Exception as e:
        raise gr.Error(
            f"We're sorry, but login failed with an error: {str(e)}"
        )


def compress_with_session(
    session: NetsPressoSession,
    model_name: str, task: Task, model_path: Union[Path, str],
    batch_size: int, channels: int, height: int, width: int,
    compression_ratio: float,
    compressed_model_path: Optional[Union[Path, str]]
) -> List[Union[NetsPressoSession, str]]:
    try:
        output_path = session.compress(
            model_name=model_name, task=task, model_path=model_path,
            batch_size=batch_size, channels=channels, height=height, width=width,
            compression_ratio=compression_ratio,
            compressed_model_path=compressed_model_path
        )
        return [session, output_path]
    except Exception as e:
        raise gr.Error(
            f"Error while compressing the model with NetsPresso: {str(e)}"
        )


def launch_gradio(args):
    with gr.Blocks(theme='nota-ai/theme', title="NetsPresso Trainer") as demo:
        gr.Markdown("\n\n# <center>Welcome to NetsPresso Trainer!</center>\n\n")
        gr.Markdown(
            "<center>Package version: "
            f"<code>netspresso-trainer=={__version__netspresso_trainer}</code> "
            f"<code>netspresso=={__version__netspresso}</code></center>"
        )

        with gr.Tab("Train"):
            gr.Markdown("\n\n### <center>TBD</center>\n\n")

        with gr.Tab("Augmentation"):
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
                with gr.Column(scale=1):
                    config_input = gr.Code(label="Augmentation configuration",
                                           value=Path(PATH_AUG_EXAMPLE_CONFIG).read_text(),
                                           language='yaml', lines=30)
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
            with gr.Row(equal_height=True):
                with gr.Column():
                    config_input = gr.Code(label="Training configuration",
                                           value=Path(PATH_SCHEDULER_EXAMPLE_CONFIG).read_text(),
                                           language='yaml', lines=30)
                    run_button = gr.Button(value="Run", variant='primary')
                with gr.Column():
                    plot_output = gr.LinePlot()

            run_button.click(get_lr_dataframe_from_config,
                             inputs=config_input, outputs=plot_output)

        with gr.Tab("PyNetsPresso"):

            session = gr.State(NetsPressoSession())

            gr.Markdown(
                "\n\n### <center>NOTE: This feature needs an internet connection</center>\n\n")

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown(
                        "If you have not signed up at NetsPresso, please sign up frist: [netspresso.ai](https://netspresso.ai/signup)")
                    email_input = gr.Textbox(
                        label="Email", type="email"
                    )
                    password_input = gr.Textbox(
                        label="Password", type="password"
                    )
                    with gr.Row(equal_height=True):
                        gr.ClearButton([email_input, password_input])
                        login_button = gr.Button(
                            value="Login", variant='primary'
                        )

                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            model_name = gr.Textbox(label="Model name")
                            model_task = gr.Dropdown(label="Task", value='classification', multiselect=False,
                                                     choices=SUPPORTING_TASK_LIST)
                        model_path = gr.Textbox(label="Model path")
                        with gr.Row():
                            compress_input_batch_size = gr.Number(label="Batch size", value=1, minimum=1, maximum=1)
                            compress_input_channels = gr.Number(label="Channels", value=3, minimum=1)
                            compress_input_height = gr.Number(label="Height", value=256, minimum=32, maximum=512)
                            compress_input_width = gr.Number(label="Width", value=256, minimum=32, maximum=512)
                        compression_ratio = gr.Slider(
                            minimum=0, maximum=1, value=0.5, step=0.1,
                            info="The removal ratio of the filters (e.g. 0.2 removes 20% of the filters in the model)"
                        )
                        compressed_model_path = gr.Textbox(label="Output model path")

                    compress_button = gr.Button(
                        value="Compress", variant='primary'
                    )
                    result_compressed_model_path = gr.Textbox(label="Result")

            login_button.click(
                login_with_session, inputs=[session, email_input, password_input], outputs=[session]
            )
            password_input.submit(
                login_with_session, inputs=[session, email_input, password_input], outputs=[session]
            )

            compress_button.click(
                compress_with_session,
                inputs=[session, model_name, model_task, model_path,
                        compress_input_batch_size, compress_input_channels, compress_input_height, compress_input_width,
                        compression_ratio, compressed_model_path],
                outputs=[session, result_compressed_model_path])

    demo.queue()

    if args.local:
        demo.launch(server_name="0.0.0.0", server_port=args.port)
    else:
        demo.launch()


if __name__ == "__main__":
    args = parse_args()
    launch_gradio(args)
