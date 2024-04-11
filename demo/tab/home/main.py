import time
from functools import partial
from pathlib import Path

import gradio as gr
from func.main import CONFIG_MODEL_DICT, DEFAULT_MODEL_DICT, load_augmentation_config, load_model_config

from tab.home.augmentation import tab_augmentation
from tab.home.dataset import tab_dataset
from tab.home.model import tab_model
from tab.home.scheduler import tab_scheduler
from tab.home.train import tab_train

# TODO: directly import from netspresso_trainer.models
SUPPORTING_TASK_LIST = ['classification', 'segmentation']
DEFAULT_TASK_NAME = 'classification'


def change_tab_to(destination=None):
    return gr.Tabs.update(selected=destination)


def copy_config(input_config):
    return input_config


def change_tab_to_train():
    return change_tab_to(destination='home-train')


def change_tab_to_dataset():
    return change_tab_to(destination='home-dataset')


def change_tab_to_augmentation():
    return change_tab_to(destination='home-augmentation')


def change_tab_to_scheduler():
    return change_tab_to(destination='home-scheduler')


def load_augmentation_config_both(task_choices):
    augmentation_config = load_augmentation_config(task=task_choices)
    return augmentation_config, augmentation_config


def tab_home(args):
    with gr.Row(equal_height=True):
        task_choices = gr.Dropdown(
            label="Task: ", value=DEFAULT_TASK_NAME, choices=SUPPORTING_TASK_LIST
        )
        model_choices = gr.Dropdown(
            label="Model: ", value=DEFAULT_MODEL_DICT[DEFAULT_TASK_NAME],
            choices=list(CONFIG_MODEL_DICT[DEFAULT_TASK_NAME].keys())
        )

    with gr.Tabs() as tabs_home:
        with gr.Tab("Train", id='home-train'):
            gr.Markdown("\n\n### <center>TBD</center>\n\n")
            train_config_dataset, train_button_dataset, train_config_augmentation, train_button_augmentation, \
                train_config_scheduler, train_button_scheduler, train_config_model, train_button_model = \
                tab_train(args, task_choices, model_choices)

        with gr.Tab("Dataset", id='home-dataset'):
            gr.Markdown("\n\n### <center>TBD</center>\n\n")
            tab_dataset(args, task_choices, model_choices)

        with gr.Tab("Augmentation", id='home-augmentation'):
            augmentation_config_input, augmentation_config_copy_button, augmentation_go_back_button = \
                tab_augmentation(args, task_choices, model_choices)

        with gr.Tab("Scheduler", id='home-scheduler'):
            scheduler_config_input, scheduler_config_copy_button, scheduler_go_back_button = \
                tab_scheduler(args, task_choices, model_choices)

    task_choices.change(
        fn=lambda t: model_choices.update(value=DEFAULT_MODEL_DICT[t],
                                          choices=list(CONFIG_MODEL_DICT[t].keys())),
        inputs=task_choices, outputs=[model_choices]
    ).success(
        fn=load_model_config,
        inputs=[task_choices, model_choices],
        outputs=[train_config_model]
    ).success(
        fn=load_augmentation_config_both,
        inputs=task_choices,
        outputs=[train_config_augmentation, augmentation_config_input]
    )

    model_choices.change(
        fn=load_model_config, inputs=[task_choices, model_choices], outputs=[train_config_model]
    )

    train_button_dataset.click(
        fn=change_tab_to_dataset, inputs=None, outputs=[tabs_home]
    )
    train_button_augmentation.click(
        fn=change_tab_to_augmentation, inputs=None, outputs=[tabs_home]
    )
    train_button_scheduler.click(
        fn=change_tab_to_scheduler, inputs=None, outputs=[tabs_home]
    )

    augmentation_config_copy_button.click(
        fn=copy_config, inputs=augmentation_config_input, outputs=train_config_augmentation
    ).success(
        fn=lambda: gr.Info("[Augmentation] Setting copied to Train tab!"), inputs=None, outputs=None
    )

    scheduler_config_copy_button.click(
        fn=copy_config, inputs=scheduler_config_input, outputs=train_config_scheduler
    ).success(
        fn=lambda: gr.Info("[Scheduler] Setting copied to Train tab!"), inputs=None, outputs=None
    )

    augmentation_go_back_button.click(
        fn=change_tab_to_train, inputs=None, outputs=[tabs_home]
    )
    scheduler_go_back_button.click(
        fn=change_tab_to_train, inputs=None, outputs=[tabs_home]
    )

    return
