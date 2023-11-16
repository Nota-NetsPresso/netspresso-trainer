import os
from pathlib import Path

import gradio as gr

PATH_SCHEDULER_DOCS = os.getenv(
    "PATH_SCHEDULER_DOCS", default="docs/description_lr_scheduler.md")
PATH_SCHEDULER_EXAMPLE_CONFIG = os.getenv(
    "PATH_SCHEDULER_EXAMPLE_CONFIG", default="config/training_template_common.yaml")


def tab_scheduler(args):
    gr.Markdown(Path(PATH_SCHEDULER_DOCS).read_text())
    with gr.Row(equal_height=True):
        with gr.Column():
            config_input = gr.Code(label="Training configuration",
                                value=Path(PATH_SCHEDULER_EXAMPLE_CONFIG).read_text(),
                                language='yaml', lines=30)
            run_button = gr.Button(value="Run", variant='primary')
        with gr.Column():
            plot_output = gr.LinePlot()
            
    return config_input, plot_output, run_button