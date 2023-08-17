
from omegaconf import OmegaConf

from .registry import SCHEDULER_DICT


def build_scheduler(optimizer, conf_training):
    scheduler_name = conf_training.sched
    num_epochs = conf_training.epochs
    conf_sched = OmegaConf.create({
        'min_lr': conf_training.min_lr,
        'power': conf_training.sched_power,
        'warmup_bias_lr': conf_training.warmup_bias_lr,
        'warmup_iters': conf_training.warmup_epochs,
        'total_iters': num_epochs,
        'iters_per_phase': conf_training.iters_per_phase,  # TODO: config for StepLR
    })
    
    assert scheduler_name in SCHEDULER_DICT, f"{scheduler_name} not in scheduler dict!"
    lr_scheduler = SCHEDULER_DICT[scheduler_name](optimizer, **conf_sched)
    
    return lr_scheduler, num_epochs
