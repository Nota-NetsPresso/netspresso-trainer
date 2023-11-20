from pathlib import Path
from functools import partial
import gradio as gr

def change_tab_to(destination=None):
    return gr.Tabs.update(selected=destination)

change_tab_to_dataset = partial(change_tab_to, destination='home-dataset')
change_tab_to_augmentation = partial(change_tab_to, destination='home-augmentation')
change_tab_to_scheduler = partial(change_tab_to, destination='home-scheduler')
change_tab_to_model = partial(change_tab_to, destination='home-model')


def tab_train(args, tabs_home):
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            train_config_dataset = gr.Code(label="Dataset configuration",
                                     language='yaml', lines=30)
            train_button_dataset = gr.Button(value="Edit in Dataset", variant='secondary')
            
        with gr.Column(scale=1):
            train_config_augmentation = gr.Code(label="Augmentation configuration",
                                          language='yaml', lines=30)
            train_button_augmentation = gr.Button(value="Edit in Augmentation", variant='secondary')
        with gr.Column(scale=1):
            train_config_scheduler = gr.Code(label="Scheduler configuration",
                                       language='yaml', lines=30)
            train_button_scheduler = gr.Button(value="Edit in Scheduler", variant='secondary')

        with gr.Column(scale=1):
            train_config_model = gr.Code(label="Model configuration",
                                   language='yaml', lines=30)
            train_button_model = gr.Button(value="Edit in Model", variant='secondary')

            
    train_button_dataset.click(
        fn=change_tab_to_dataset, inputs=None, outputs=[tabs_home]
    )
    train_button_augmentation.click(
        fn=change_tab_to_augmentation, inputs=None, outputs=[tabs_home]
    )
    train_button_scheduler.click(
        fn=change_tab_to_scheduler, inputs=None, outputs=[tabs_home]
    )
    train_button_model.click(
        fn=change_tab_to_model, inputs=None, outputs=[tabs_home]
    )