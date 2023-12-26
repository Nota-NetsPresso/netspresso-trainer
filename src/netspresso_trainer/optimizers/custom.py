import torch.optim as optim

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