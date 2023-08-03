""" Scheduler Factory
Hacked together by / Copyright 2021 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
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
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(conf_sched, 'lr_noise_pct', 0.67),
        noise_std=getattr(conf_sched, 'lr_noise_std', 1.),
        noise_seed=getattr(conf_sched, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(conf_sched, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(conf_sched, 'lr_cycle_decay', 0.5),
        cycle_limit=getattr(conf_sched, 'lr_cycle_limit', 1),
    )

    lr_scheduler = None
    if getattr(conf_sched, 'sched') == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=conf_sched.min_lr,
            warmup_lr_init=conf_sched.warmup_lr,
            warmup_t=conf_sched.warmup_epochs,
            k_decay=getattr(conf_sched, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + conf_sched.cooldown_epochs
    elif getattr(conf_sched, 'sched') == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=conf_sched.min_lr,
            warmup_lr_init=conf_sched.warmup_lr,
            warmup_t=conf_sched.warmup_epochs,
            t_in_epochs=True,
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + conf_sched.cooldown_epochs
    elif getattr(conf_sched, 'sched') == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=conf_sched.decay_epochs,
            decay_rate=conf_sched.decay_rate,
            warmup_lr_init=conf_sched.warmup_lr,
            warmup_t=conf_sched.warmup_epochs,
            **noise_args,
        )
    elif getattr(conf_sched, 'sched') == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=conf_sched.decay_milestones,
            decay_rate=conf_sched.decay_rate,
            warmup_lr_init=conf_sched.warmup_lr,
            warmup_t=conf_sched.warmup_epochs,
            **noise_args,
        )
    elif getattr(conf_sched, 'sched') == 'plateau':
        mode = 'min' if 'loss' in getattr(conf_sched, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=conf_sched.decay_rate,
            patience_t=conf_sched.patience_epochs,
            lr_min=conf_sched.min_lr,
            mode=mode,
            warmup_lr_init=conf_sched.warmup_lr,
            warmup_t=conf_sched.warmup_epochs,
            cooldown_t=0,
            **noise_args,
        )
    elif getattr(conf_sched, 'sched') == 'poly':
        # lr_scheduler = PolyLRScheduler(
        #     optimizer,
        #     power=conf_sched.decay_rate,  # overloading 'decay_rate' as polynomial power
        #     t_initial=num_epochs,
        #     lr_min=conf_sched.min_lr,
        #     warmup_lr_init=conf_sched.warmup_lr,
        #     warmup_t=conf_sched.warmup_epochs,
        #     k_decay=getattr(conf_sched, 'lr_k_decay', 1.0),
        #     **cycle_args,
        #     **noise_args,
        # )
        # num_epochs = lr_scheduler.get_cycle_length() + conf_sched.cooldown_epochs
        
        lr_scheduler = PolynomialLRWithWarmUp(
            optimizer,
            warmup_iters=conf_sched.warmup_epochs,
            total_iters=num_epochs,
            power=conf_sched.decay_rate
        )

    return lr_scheduler, num_epochs
