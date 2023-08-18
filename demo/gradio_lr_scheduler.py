from typing import Tuple, List, Union
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import netspresso_trainer
from netspresso_trainer.schedulers import SCHEDULER_DICT

__version__ = netspresso_trainer.__version__

CURRENT_DIR = Path(__file__).resolve().parent


def get_lr_dataframe_from_config(yaml_str: str):
    config = OmegaConf.create(yaml_str)


def launch_gradio(args):
    with gr.Blocks(theme='nota-ai/theme', title="LR Scheduler Simulator with NetsPresso Trainer") as demo:
        gr.Markdown((CURRENT_DIR / "docs" / "description_lr_scheduler.md").read_text())
        gr.Markdown(f"<center>Package version: <code>netspresso-trainer-{__version__}</code></center>")
        with gr.Row():
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
