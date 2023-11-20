from pathlib import Path
from functools import partial
import gradio as gr
import time

from tab.home.train import tab_train
from tab.home.dataset import tab_dataset
from tab.home.augmentation import tab_augmentation
from tab.home.scheduler import tab_scheduler
from tab.home.model import tab_model

def change_tab_to(destination=None):
    return gr.Tabs.update(selected=destination)

def copy_config(input_config):
    return input_config

change_tab_to_train = lambda: change_tab_to(destination='home-train')
change_tab_to_dataset = lambda: change_tab_to(destination='home-dataset')
change_tab_to_augmentation = lambda: change_tab_to(destination='home-augmentation')
change_tab_to_scheduler =lambda: change_tab_to(destination='home-scheduler')
change_tab_to_model = lambda: change_tab_to(destination='home-model')

def tab_home(args):
    with gr.Tabs() as tabs_home:
        with gr.Tab("Train", id='home-train'):
            gr.Markdown("\n\n### <center>TBD</center>\n\n")
            train_config_dataset, train_button_dataset, train_config_augmentation, train_button_augmentation, \
                train_config_scheduler, train_button_scheduler, train_config_model, train_button_model = \
                tab_train(args)
            
        with gr.Tab("Dataset", id='home-dataset'):
            gr.Markdown("\n\n### <center>TBD</center>\n\n")
            tab_dataset(args)

        with gr.Tab("Augmentation", id='home-augmentation'):
            augmentation_config_input, augmentation_config_copy_button, augmentation_go_back_button = tab_augmentation(args)

        with gr.Tab("Scheduler", id='home-scheduler'):
            scheduler_config_input, scheduler_config_copy_button, scheduler_go_back_button = tab_scheduler(args)
            
        with gr.Tab("Model", id='home-model'):
            gr.Markdown("\n\n### <center>TBD</center>\n\n")
            tab_model(args)
            
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