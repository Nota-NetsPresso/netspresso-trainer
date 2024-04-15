import os
from pathlib import Path

import gradio as gr
from func.scheduler import get_lr_dataframe_from_config

PATH_SCHEDULER_DOCS = os.getenv(
    "PATH_SCHEDULER_DOCS", default="docs/description_scheduler.md")
PATH_CONFIG_ROOT = os.getenv("PATH_CONFIG_ROOT", default="config/")
PATH_SCHEDULER_EXAMPLE_CONFIG = Path(PATH_CONFIG_ROOT) / "training.yaml"


def tab_scheduler(args, task_choices, model_choices):
    gr.Markdown(Path(PATH_SCHEDULER_DOCS).read_text())
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
                config_input = gr.Code(label="Training configuration",
                                       value=Path(PATH_SCHEDULER_EXAMPLE_CONFIG).read_text(),
                                       language='yaml', lines=15)
                with gr.Row():
                    go_back_button = gr.Button(value="Back to Train", variant='secondary')
                    config_copy_button = gr.Button(value="Copy to Train", variant='secondary')
            run_button = gr.Button(value="Run", variant='primary')
        with gr.Column(scale=2):
            plot_output = gr.LinePlot()

        run_button.click(get_lr_dataframe_from_config, inputs=config_input, outputs=plot_output)

    return config_input, config_copy_button, go_back_button
