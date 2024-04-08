import os
from functools import partial
from pathlib import Path

import gradio as gr
from func.main import load_model_config

PATH_CONFIG_ROOT = os.getenv("PATH_CONFIG_ROOT", default="config/")
PATH_AUG_EXAMPLE_CONFIG = Path(PATH_CONFIG_ROOT) / "augmentation/classification.yaml"
PATH_SCHEDULER_EXAMPLE_CONFIG = Path(PATH_CONFIG_ROOT) / "training.yaml"


def change_tab_to(destination=None):
    return gr.Tabs.update(selected=destination)


def tab_train(args, task_choices, model_choices):
    with gr.Row(equal_height=True):

        with gr.Column(scale=2), gr.Group():
            train_config_model = gr.Code(label="Model configuration",
                                         value=load_model_config(task=task_choices.value, model=model_choices.value),
                                         language='yaml', lines=42,
                                         interactive=True)
            train_button_model = gr.Button(value="Edit in Model", variant='secondary', visible=False)

        with gr.Column(scale=2), gr.Group():
            train_config_dataset = gr.Code(label="Dataset configuration",
                                           value="TBD",
                                           language='yaml', lines=40)
            train_button_dataset = gr.Button(value="Edit in Dataset", variant='secondary')

        with gr.Column(scale=1):
            with gr.Group():
                train_config_augmentation = gr.Code(label="Augmentation configuration",
                                                    value=Path(PATH_AUG_EXAMPLE_CONFIG).read_text(),
                                                    language='yaml', lines=15)
                train_button_augmentation = gr.Button(value="Edit in Augmentation", variant='secondary')
            with gr.Group():
                train_config_scheduler = gr.Code(label="Scheduler configuration",
                                                 value=Path(PATH_SCHEDULER_EXAMPLE_CONFIG).read_text(),
                                                 language='yaml', lines=15)
                train_button_scheduler = gr.Button(value="Edit in Scheduler", variant='secondary')

    return train_config_dataset, train_button_dataset, train_config_augmentation, train_button_augmentation, \
        train_config_scheduler, train_button_scheduler, train_config_model, train_button_model
