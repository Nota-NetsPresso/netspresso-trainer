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

import math
import random
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms.functional as F
from torch.nn import functional as F_torch
from torchvision.transforms.functional import InterpolationMode


class Mixing:
    visualize = False

    def __init__(
        self,
        num_classes: int,
        cutmix: Optional[List],
        mixup: Optional[List],
        inplace: bool,
    ):
        self.mixup = bool(mixup)
        self.cutmix = bool(cutmix)
        self.num_classes = num_classes
        assert self.mixup or self.cutmix, "One of mixup or cutmix must be activated."

        self.transforms = []
        if self.mixup:
            assert len(mixup) == 2, "Mixup transform definition must be List of length 2."
            self.mixup_alpha, self.mixup_p = mixup
            self.transforms.append(RandomMixup(num_classes, self.mixup_alpha, self.mixup_p, inplace))

        if self.cutmix:
            assert len(cutmix) == 2, "Cutmix transform definition must be List of length 2."
            self.cutmix_alpha, self.cutmix_p = cutmix
            self.transforms.append(RandomCutmix(num_classes, self.cutmix_alpha, self.cutmix_p, inplace))

    def __call__(self, samples, targets):
        _mixup_transform = random.choice(self.transforms)
        return _mixup_transform(samples, targets)

    def __repr__(self) -> str:
        repr = "{}(num_classes={}, ".format(self.__class__.__name__, self.num_classes)
        if self.mixup:
            repr += "mixup_p={}, mixup_alpha={}, ".format(self.mixup_p, self.mixup_alpha)
        if self.cutmix:
            repr += "cutmix_p={}, alpha={}, ".format(self.cutmix_p, self.cutmix_alpha)
        repr += "inplace={})".format(self.inplace)
        return repr


class RandomMixup:
    """
    Based on the RandomMixup implementation of ml_cvnets.
    https://github.com/apple/ml-cvnets/blob/77717569ab4a852614dae01f010b32b820cb33bb/data/transforms/image_torch.py

    Given a batch of input images and labels, this class randomly applies the
    `MixUp transformation <https://arxiv.org/abs/1710.09412>`_

    Args:
        opts (argparse.Namespace): Arguments
        num_classes (int): Number of classes in the dataset
    """
    visualize = False

    def __init__(
        self,
        num_classes: int,
        alpha: float,
        p: float,
        inplace: bool,
    ):
        if not (num_classes > 0):
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if not (alpha > 0):
            raise ValueError("Alpha param can't be zero.")
        if not (0.0 < p <= 1.0):
            raise ValueError("MixUp probability should be between 0 and 1, where 1 is inclusive")

        self.num_classes = num_classes
        self.alpha = alpha
        self.p = p
        self.inplace = inplace

    def _apply_mixup_transform(self, image_tensor, target_tensor):
        if image_tensor.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {image_tensor.ndim}")
        if target_tensor.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target_tensor.ndim}")
        if not image_tensor.is_floating_point():
            raise ValueError(f"Batch datatype should be a float tensor. Got {image_tensor.dtype}.")
        if target_tensor.dtype != torch.int64:
            raise ValueError(f"Target datatype should be torch.int64. Got {target_tensor.dtype}")

        if not self.inplace:
            image_tensor = image_tensor.clone()
            target_tensor = target_tensor.clone()

        if target_tensor.ndim == 1:
            target_tensor = F_torch.one_hot(
                target_tensor, num_classes=self.num_classes
            ).to(dtype=image_tensor.dtype)

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = image_tensor.roll(1, 0)
        target_rolled = target_tensor.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        image_tensor.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target_tensor.mul_(lambda_param).add_(target_rolled)
        return image_tensor, target_tensor

    def __call__(self, samples, targets):
        if torch.rand(1).item() >= self.p:
            return samples, targets

        mixup_samples, mixup_targets = self._apply_mixup_transform(
            image_tensor=samples, target_tensor=targets
        )

        return mixup_samples, mixup_targets

    def __repr__(self) -> str:
        return "{}(num_classes={}, p={}, alpha={}, inplace={})".format(
            self.__class__.__name__, self.num_classes, self.p, self.alpha, self.inplace
        )


class RandomCutmix:
    """
    Based on the RandomCutmix implementation of ml_cvnets.
    https://github.com/apple/ml-cvnets/blob/77717569ab4a852614dae01f010b32b820cb33bb/data/transforms/image_torch.py

    Given a batch of input images and labels, this class randomly applies the
    `CutMix transformation <https://arxiv.org/abs/1905.04899>`_

    Args:
        opts (argparse.Namespace): Arguments
        num_classes (int): Number of classes in the dataset
    """
    visualize = False

    def __init__(
        self,
        num_classes: int,
        alpha: float,
        p: float,
        inplace: bool,
    ):
        if not (num_classes > 0):
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if not (alpha > 0):
            raise ValueError("Alpha param can't be zero.")
        if not (0.0 < p <= 1.0):
            raise ValueError("CutMix probability should be between 0 and 1, where 1 is inclusive")

        self.num_classes = num_classes
        self.alpha = alpha
        self.p = p
        self.inplace = inplace

    def _apply_cutmix_transform(self, image_tensor, target_tensor):
        if image_tensor.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {image_tensor.ndim}")
        if target_tensor.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target_tensor.ndim}")
        if not image_tensor.is_floating_point():
            raise ValueError(f"Batch dtype should be a float tensor. Got {image_tensor.dtype}.")
        if target_tensor.dtype != torch.int64:
            raise ValueError(f"Target dtype should be torch.int64. Got {target_tensor.dtype}")

        if not self.inplace:
            image_tensor = image_tensor.clone()
            target_tensor = target_tensor.clone()

        if target_tensor.ndim == 1:
            target_tensor = F_torch.one_hot(
                target_tensor, num_classes=self.num_classes
            ).to(dtype=image_tensor.dtype)

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = image_tensor.roll(1, 0)
        target_rolled = target_tensor.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        W, H = F.get_image_size(image_tensor)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        image_tensor[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target_tensor.mul_(lambda_param).add_(target_rolled)
        return image_tensor, target_tensor

    def __call__(self, samples, targets) -> Dict:
        if torch.rand(1).item() >= self.p:
            return samples, targets

        mixup_samples, mixup_targets = self._apply_cutmix_transform(
            image_tensor=samples, target_tensor=targets
        )

        return mixup_samples, mixup_targets

    def __repr__(self) -> str:
        return "{}(num_classes={}, p={}, alpha={}, inplace={})".format(
            self.__class__.__name__, self.num_classes, self.p, self.alpha, self.inplace
        )

