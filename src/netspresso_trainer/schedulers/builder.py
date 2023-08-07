""" Scheduler Factory
Hacked together by / Copyright 2021 Ross Wightman
"""
from .cosine_lr import CosineAnnealingLRWithCustomWarmUp
from .multistep_lr import MultiStepLRScheduler
from .plateau_lr import PlateauLRScheduler
from .poly_lr import PolynomialLRWithWarmUp
from .step_lr import StepLRScheduler
from .tanh_lr import TanhLRScheduler


def build_scheduler(optimizer, conf_sched):
    num_epochs = getattr(conf_sched, 'epochs')

    if getattr(conf_sched, 'lr_noise', None) is not None:
        lr_noise = getattr(conf_sched, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None
    if getattr(conf_sched, 'sched') == 'cosine':
        lr_scheduler = CosineAnnealingLRWithCustomWarmUp(
            optimizer,
            warmup_iters=conf_sched.warmup_epochs,
            total_iters=num_epochs,
            lr_min=conf_sched.min_lr
        )
    elif getattr(conf_sched, 'sched') == 'poly':
        lr_scheduler = PolynomialLRWithWarmUp(
            optimizer,
            warmup_iters=conf_sched.warmup_epochs,
            total_iters=num_epochs,
            power=conf_sched.decay_rate
        )

    return lr_scheduler, num_epochs
