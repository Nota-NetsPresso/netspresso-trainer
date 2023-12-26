
from omegaconf import OmegaConf

from .registry import SCHEDULER_DICT


def build_scheduler(optimizer, conf_training):
    conf_scheduler = conf_training.scheduler
    scheduler_name = conf_scheduler.name
    num_epochs = conf_training.epochs

    # Copy training num_epochs to sub-config scheduler
    conf_scheduler.total_iters = num_epochs # fix total_iters as num_epochs

    assert scheduler_name in SCHEDULER_DICT, f"{scheduler_name} not in scheduler dict!"
    lr_scheduler = SCHEDULER_DICT[scheduler_name](optimizer, conf_scheduler)

    return lr_scheduler, num_epochs
