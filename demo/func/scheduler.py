import os
from typing import List

import gradio as gr
import pandas as pd
import torch
from netspresso_trainer.optimizers import build_optimizer
from netspresso_trainer.schedulers import build_scheduler
from omegaconf import OmegaConf


def _get_lr_list(
        optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, total_epochs: int) -> List[
        float]:
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
                                    optimizer_conf=conf.training.optimizer)
        scheduler, total_epochs = build_scheduler(optimizer, conf.training)
        lr_list = _get_lr_list(optimizer, scheduler, total_epochs)

        df = pd.DataFrame({
            "lr": lr_list,
            "epochs": list(range(1, total_epochs + 1))
        })
        return gr.LinePlot(df, x="epochs", y="lr", width=600, height=300, tooltip=["epochs", "lr"])
    except Exception as e:
        raise gr.Error(str(e)) from e
