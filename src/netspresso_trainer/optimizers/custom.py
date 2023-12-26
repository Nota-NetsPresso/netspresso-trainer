import torch.optim as optim


class Adam(optim.Adam):
    def __init__(
        self,
        params,
        optimizer_conf,
    ) -> None:
        lr = optimizer_conf.lr
        betas = optimizer_conf.betas
        weight_decay = optimizer_conf.weight_decay
        
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)


class AdamW(optim.AdamW):
    def __init__(
        self,
        params,
        optimizer_conf,
    ) -> None:
        lr = optimizer_conf.lr
        betas = optimizer_conf.betas
        weight_decay = optimizer_conf.weight_decay
        
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)


class RMSprop(optim.RMSprop):
    def __init__(
        self,
        params,
        optimizer_conf,
    ) -> None:
        lr = optimizer_conf.lr
        alpha = optimizer_conf.alpha
        weight_decay = optimizer_conf.weight_decay
        momentum = optimizer_conf.momentum

        super().__init__(params=params, lr=lr, alpha=alpha, weight_decay=weight_decay, momentum=momentum)


class SGD(optim.SGD):
    def __init__(
        self,
        params,
        optimizer_conf,
    ) -> None:
        lr = optimizer_conf.lr
        momentum = optimizer_conf.momentum
        weight_decay = optimizer_conf.weight_decay
        nesterov = optimizer_conf.nesterov

        super().__init__(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)