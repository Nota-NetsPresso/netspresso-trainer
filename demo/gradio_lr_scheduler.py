from typing import Tuple, List, Union
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import torch

import netspresso_trainer
from netspresso_trainer.optimizers import build_optimizer
from netspresso_trainer.schedulers import build_scheduler
from netspresso_trainer.loggers import START_EPOCH_ZERO_OR_ONE

__version__ = netspresso_trainer.__version__

CURRENT_DIR = Path(__file__).resolve().parent


def get_lr_list(scheduler: torch.optim.lr_scheduler._LRScheduler, total_epochs: int) -> List[float]:
    lr_list = []
    for epoch in range(total_epochs):
        lr = scheduler.get_lr()[0]
        lr_list.append(lr)
        scheduler.step()
    return lr_list


def get_lr_dataframe_from_config(yaml_str: str):

    conf = OmegaConf.create(yaml_str)
    model_mock = torch.nn.Linear(1, 1)
    optimizer = build_optimizer(model_mock,
                                opt=conf.training.opt,
                                lr=conf.training.lr,
                                wd=conf.training.weight_decay,
                                momentum=conf.training.momentum)
    scheduler, total_epochs = build_scheduler(optimizer, conf.training)
    lr_list = get_lr_list(scheduler, total_epochs)

    df = pd.DataFrame(dict(
        lr=lr_list,
        epochs=list(range(START_EPOCH_ZERO_OR_ONE, total_epochs + START_EPOCH_ZERO_OR_ONE))
    ))

    return gr.LinePlot.update(df, x="epochs", y="lr", width=600, height=300)


def launch_gradio(args):
    with gr.Blocks(theme='nota-ai/theme', title="LR Scheduler Simulator with NetsPresso Trainer") as demo:
        gr.Markdown((CURRENT_DIR / "docs" / "description_lr_scheduler.md").read_text())
        gr.Markdown(f"<center>Package version: <code>netspresso-trainer-{__version__}</code></center>")
        with gr.Row().style(equal_height=True):
            with gr.Column():
                example_training_config_path = CURRENT_DIR.parent / "config" / "training" / "template" / "common.yaml"
                config_input = gr.Code(label="Training configuration", value=example_training_config_path.read_text(), language='yaml', lines=30)
                run_button = gr.Button(value="Run", variant='primary')
            with gr.Column():
                plot_output = gr.LinePlot()

        run_button.click(get_lr_dataframe_from_config, inputs=config_input, outputs=plot_output)

    demo.launch(server_name="0.0.0.0", server_port=50002)


if __name__ == "__main__":
    launch_gradio(args=None)
