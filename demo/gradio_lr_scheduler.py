import argparse
import os
from pathlib import Path
from typing import List, Tuple, Union

import gradio as gr
import netspresso_trainer
import numpy as np
import pandas as pd
import torch
from netspresso_trainer.optimizers import build_optimizer
from netspresso_trainer.schedulers import build_scheduler
from omegaconf import OmegaConf

__version__ = netspresso_trainer.__version__

PATH_DOCS = os.getenv("PATH_DOCS", default="docs/description_lr_scheduler.md")
PATH_EXAMPLE_CONFIG = os.getenv("PATH_EXAMPLE_CONFIG", default="config/training_template_common.yaml")


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
            "epochs": list(range(1, total_epochs + 1))
        })
        return gr.LinePlot.update(df, x="epochs", y="lr", width=600, height=300, tooltip=["epochs", "lr"])
    except Exception as e:
        raise gr.Error(str(e))


def parse_args():

    parser = argparse.ArgumentParser(description="LR scheduler simulator for NetsPresso Trainer")

    # -------- User arguments ----------------------------------------

    parser.add_argument(
        '--docs', type=Path, default=PATH_DOCS,
        help="Docs string file")

    parser.add_argument(
        '--config', type=Path, default=PATH_EXAMPLE_CONFIG,
        help="Config for lr scheduler")

    parser.add_argument(
        '--local', action='store_true',
        help="Whether to run in local environment or not")

    parser.add_argument(
        '--port', type=int, default=50002,
        help="Service port (only applicable when running on local server)")

    args, _ = parser.parse_known_args()

    return args


def launch_gradio(args):
    with gr.Blocks(theme='nota-ai/theme', title="LR Scheduler Simulator with NetsPresso Trainer") as demo:
        gr.Markdown(args.docs.read_text())
        gr.Markdown(f"<center>Package version: <code>netspresso-trainer-{__version__}</code></center>")
        with gr.Row(equal_height=True):
            with gr.Column():
                config_input = gr.Code(label="Training configuration", value=args.config.read_text(), language='yaml', lines=30)
                run_button = gr.Button(value="Run", variant='primary')
            with gr.Column():
                plot_output = gr.LinePlot()

        run_button.click(get_lr_dataframe_from_config, inputs=config_input, outputs=plot_output)

    if args.local:
        demo.launch(server_name="0.0.0.0", server_port=args.port)
    else:
        demo.launch()


if __name__ == "__main__":
    args = parse_args()
    launch_gradio(args)
