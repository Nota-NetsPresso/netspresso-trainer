
from omegaconf import OmegaConf

from .registry import SCHEDULER_DICT


def build_scheduler(optimizer, training_conf):
    scheduler_conf = training_conf.scheduler
    scheduler_name = scheduler_conf.name
    num_epochs = training_conf.epochs

    # Copy training num_epochs to sub-config scheduler
    scheduler_conf.total_iters = num_epochs # fix total_iters as num_epochs

    assert scheduler_name in SCHEDULER_DICT, f"{scheduler_name} not in scheduler dict!"
    lr_scheduler = SCHEDULER_DICT[scheduler_name](optimizer, scheduler_conf)

    return lr_scheduler, num_epochs
