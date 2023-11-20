import os
from functools import partial
from pathlib import Path

import gradio as gr

PATH_AUG_EXAMPLE_CONFIG = os.getenv(
    "PATH_AUG_EXAMPLE_CONFIG", default="config/augmentation_template_common.yaml")
PATH_SCHEDULER_EXAMPLE_CONFIG = os.getenv(
    "PATH_SCHEDULER_EXAMPLE_CONFIG", default="config/training_template_common.yaml")


def change_tab_to(destination=None):
    return gr.Tabs.update(selected=destination)




def tab_train(args):
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            train_config_dataset = gr.Code(label="Dataset configuration",
                                           value="TBD",
                                           language='yaml', lines=30)
            train_button_dataset = gr.Button(value="Edit in Dataset", variant='secondary')

        with gr.Column(scale=1):
            train_config_augmentation = gr.Code(label="Augmentation configuration",
                                                value=Path(PATH_AUG_EXAMPLE_CONFIG).read_text(),
                                                language='yaml', lines=30)
            train_button_augmentation = gr.Button(value="Edit in Augmentation", variant='secondary')
        with gr.Column(scale=1):
            train_config_scheduler = gr.Code(label="Scheduler configuration",
                                             value=Path(PATH_SCHEDULER_EXAMPLE_CONFIG).read_text(),
                                             language='yaml', lines=30)
            train_button_scheduler = gr.Button(value="Edit in Scheduler", variant='secondary')

        with gr.Column(scale=1):
            train_config_model = gr.Code(label="Model configuration",
                                         value="TBD",
                                         language='yaml', lines=30)
            train_button_model = gr.Button(value="Edit in Model", variant='secondary')

    return train_config_dataset, train_button_dataset, train_config_augmentation, train_button_augmentation, \
        train_config_scheduler, train_button_scheduler, train_config_model, train_button_model


