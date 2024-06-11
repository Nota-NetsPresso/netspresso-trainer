# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import torch.optim as optim


class Adadelta(optim.Adadelta):
    def __init__(
        self,
        params,
        optimizer_conf,
    ) -> None:
        lr = optimizer_conf.lr
        rho = optimizer_conf.rho
        weight_decay = optimizer_conf.weight_decay

        super().__init__(params=params, lr=lr, rho=rho, weight_decay=weight_decay)


class Adagrad(optim.Adagrad):
    def __init__(
        self,
        params,
        optimizer_conf,
    ) -> None:
        lr = optimizer_conf.lr
        lr_decay = optimizer_conf.lr_decay
        weight_decay = optimizer_conf.weight_decay

        super().__init__(params=params, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)


class Adam(optim.Adam):
    def __init__(
        self,
        params,
        optimizer_conf,
    ) -> None:
        lr = optimizer_conf.lr
        betas = optimizer_conf.betas
        weight_decay = optimizer_conf.weight_decay

        super().__init__(params=params, lr=lr, betas=betas, weight_decay=weight_decay)


class Adamax(optim.Adamax):
    def __init__(
        self,
        params,
        optimizer_conf,
    ) -> None:
        lr = optimizer_conf.lr
        betas = optimizer_conf.betas
        weight_decay = optimizer_conf.weight_decay

        super().__init__(params=params, lr=lr, betas=betas, weight_decay=weight_decay)


class AdamW(optim.AdamW):
    def __init__(
        self,
        params,
        optimizer_conf,
    ) -> None:
        lr = optimizer_conf.lr
        betas = optimizer_conf.betas
        weight_decay = optimizer_conf.weight_decay

        super().__init__(params=params, lr=lr, betas=betas, weight_decay=weight_decay)


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
        eps = optimizer_conf.eps

        super().__init__(params=params, lr=lr, alpha=alpha, weight_decay=weight_decay, momentum=momentum, eps=eps)


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
