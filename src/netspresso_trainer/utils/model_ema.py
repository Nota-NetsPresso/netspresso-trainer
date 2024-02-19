from copy import deepcopy

import torch

__all__ = ['ModelEMA']


class ModelEMA:
    def __init__(self, model, decay, device=None):
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_param = model_param.to(device=self.device)
                updated_param = self.decay * ema_param + (1. - self.decay) * model_param
                ema_param.copy_(updated_param)
