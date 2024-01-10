import math
import random
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from omegaconf import ListConfig
from torch import Tensor
from torch.nn import functional as F_torch
from torchvision.transforms.autoaugment import _apply_op
from torchvision.transforms.functional import InterpolationMode

BBOX_CROP_KEEP_THRESHOLD = 0.2
MAX_RETRY = 5
INVERSE_MODES_MAPPING = {
    'nearest': InterpolationMode.NEAREST,
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
}


class Compose:
    def __init__(self, transforms, additional_targets: Dict = None):
        if additional_targets is None:
            additional_targets = {}
        self.transforms = transforms
        self.additional_targets = additional_targets

    def _get_transformed(self, image, mask, bbox, visualize_for_debug):
        for t in self.transforms:
            if visualize_for_debug and not t.visualize:
                continue
            image, mask, bbox = t(image=image, mask=mask, bbox=bbox)
        return image, mask, bbox

    def __call__(self, image, mask=None, bbox=None, visualize_for_debug=False, **kwargs):
        additional_targets_result = {k: None for k in kwargs if k in self.additional_targets}

        result_image, result_mask, result_bbox = self._get_transformed(image=image, mask=mask, bbox=bbox, visualize_for_debug=visualize_for_debug)
        for key in additional_targets_result:
            if self.additional_targets[key] == 'mask':
                _, additional_targets_result[key], _ = self._get_transformed(image=image, mask=kwargs[key], bbox=None, visualize_for_debug=visualize_for_debug)
            elif self.additional_targets[key] == 'bbox':
                _, _, additional_targets_result[key] = self._get_transformed(image=image, mask=None, bbox=kwargs[key], visualize_for_debug=visualize_for_debug)
            else:
                del additional_targets_result[key]

        return_dict = {'image': result_image}
        if mask is not None:
            return_dict.update({'mask': result_mask})
        if bbox is not None:
            return_dict.update({'bbox': result_bbox})
        return_dict.update(additional_targets_result)
        return return_dict

    def __repr__(self):
        compose_summary = "CustomCompose"
        compose_list = ",\n\t".join([str(t) for t in self.transforms])
        compose_summary += "(\n\t" + compose_list + "\n)"
        return compose_summary


class Identity:
    visualize = True

    def __init__(self):
        pass

    def __call__(self, image, mask=None, bbox=None):
        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Pad(T.Pad):
    visualize = True
    def __init__(
        self,
        padding: Union[int, List],
        fill: Union[int, List],
        padding_mode: str,
    ):
        super().__init__(padding, fill, padding_mode)

    def forward(self, image, mask=None, bbox=None):
        image = F.pad(image, self.padding, self.fill, self.padding_mode)
        if mask is not None:
            mask = F.pad(mask, self.padding, fill=255, padding_mode=self.padding_mode)
        if bbox is not None:
            target_padding = [self.padding] if not isinstance(self.padding, Sequence) else self.padding

            padding_left, padding_top, _, _ = \
                target_padding * (4 / len(target_padding))  # supports 1, 2, 4 length

            bbox[..., 0:4:2] += padding_left
            bbox[..., 1:4:2] += padding_top

        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + "(padding={0}, fill={1}, padding_mode={2})".format(
            self.padding, self.fill, self.padding_mode)


class Resize(T.Resize):
    visualize = True

    def __init__(
        self,
        size: int,
        interpolation: str,
        max_size: Optional[int],
    ):
        interpolation = INVERSE_MODES_MAPPING[interpolation]

        # TODO: There is logic error in forward. If `size` is int, this specify edge for shorter one.
        # And, this is not match with bbox computing logic.
        # Thus, automatically transform to sequence format for now,
        # but this should be specified whether Resize receives sequence or int.
        if isinstance(size, int):
            size = [size, size]

        # @illian01: antialias paramter always true (always use) for PIL image, and NetsPresso Trainer uses PIL image for augmentation.
        super().__init__(size, interpolation, max_size)

    def forward(self, image, mask=None, bbox=None):
        w, h = image.size

        image = F.resize(image, self.size, self.interpolation, self.max_size, self.antialias)
        if mask is not None:
            mask = F.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST,
                            max_size=self.max_size)
        if bbox is not None:
            target_w, target_h = (self.size, self.size) if isinstance(self.size, int) else self.size
            bbox[..., 0:4:2] *= float(target_w / w)
            bbox[..., 1:4:2] *= float(target_h / h)
        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, interpolation={1}, max_size={2}, antialias={3})".format(
            self.size, self.interpolation.value, self.max_size, self.antialias)


class RandomHorizontalFlip:
    visualize = True

    def __init__(
        self,
        p: float,
    ):
        self.p: float = max(0., min(1., p))

    def __call__(self, image, mask=None, bbox=None):
        w, _ = image.size
        if random.random() < self.p:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
            if bbox is not None:
                bbox[..., 2::-2] = w - bbox[..., 0:4:2]
        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)


class RandomVerticalFlip:
    visualize = True

    def __init__(
        self,
        p: float,
    ):
        self.p: float = max(0., min(1., p))

    def __call__(self, image, mask=None, bbox=None):
        _, h = image.size
        if random.random() < self.p:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)
            if bbox is not None:
                bbox[..., 3::-2] = h - bbox[..., 1:4:2]
        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)


class PadIfNeeded:
    visualize = True

    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.new_h = size[0] if isinstance(size, Sequence) else size
        self.new_w = size[1] if isinstance(size, Sequence) else size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image, mask=None, bbox=None):
        if not isinstance(image, (torch.Tensor, Image.Image)):
            raise TypeError("Image should be Tensor or PIL.Image. Got {}".format(type(image)))

        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            w, h = image.shape[-1], image.shape[-2]

        w_pad_needed = max(0, self.new_w - w)
        h_pad_needed = max(0, self.new_h - h)
        padding_ltrb = [w_pad_needed // 2,
                        h_pad_needed // 2,
                        w_pad_needed // 2 + w_pad_needed % 2,
                        h_pad_needed // 2 + h_pad_needed % 2]
        image = F.pad(image, padding_ltrb, fill=self.fill, padding_mode=self.padding_mode)
        if mask is not None:
            mask = F.pad(mask, padding_ltrb, fill=255, padding_mode=self.padding_mode)
        if bbox is not None:
            padding_left, padding_top, _, _ = padding_ltrb
            bbox[..., 0:4:2] += padding_left
            bbox[..., 1:4:2] += padding_top
        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, fill={1}, padding_mode={2})".format(
            (self.new_h, self.new_w), self.fill, self.padding_mode
        )


class ColorJitter(T.ColorJitter):
    visualize = True

    def __init__(
        self,
        brightness: Union[float, List],
        contrast: Union[float, List],
        saturation: Union[float, List],
        hue: Union[float, List],
        p: float
    ):
        super(ColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.p: float = max(0., min(1., p))

    def forward(self, image, mask=None, bbox=None):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        if random.random() < self.p:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    image = F.adjust_brightness(image, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    image = F.adjust_contrast(image, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    image = F.adjust_saturation(image, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    image = F.adjust_hue(image, hue_factor)

        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + \
            f"(brightness={self.brightness}, " + \
            f"contrast={self.contrast}, " + \
            f"saturation={self.saturation}, " + \
            f"hue={self.hue}, " + \
            f"p={self.p})"


class RandomCrop:
    visualize = True

    def __init__(
        self,
        size: Union[int, List],
    ):

        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size_h = size[0] if isinstance(size, Sequence) else size
        self.size_w = size[1] if isinstance(size, Sequence) else size
        self.image_pad_if_needed = PadIfNeeded((self.size_h, self.size_w))

    def _crop_bbox(self, bbox, i, j, h, w):
        area_original = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])

        bbox[..., 0:4:2] = np.clip(bbox[..., 0:4:2] - j, 0, w)
        bbox[..., 1:4:2] = np.clip(bbox[..., 1:4:2] - i, 0, h)

        area_cropped = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
        area_ratio = area_cropped / (area_original + 1)  # +1 for preventing ZeroDivisionError

        bbox = bbox[area_ratio >= BBOX_CROP_KEEP_THRESHOLD, ...]
        return bbox

    def __call__(self, image, mask=None, bbox=None):
        image, mask, bbox = self.image_pad_if_needed(image=image, mask=mask, bbox=bbox)
        i, j, h, w = T.RandomCrop.get_params(image, (self.size_h, self.size_w))
        image = F.crop(image, i, j, h, w)
        if mask is not None:
            mask = F.crop(mask, i, j, h, w)
        if bbox is not None:
            bbox_candidate = self._crop_bbox(bbox, i, j, h, w)
            _bbox_crop_count = 1
            while bbox_candidate.shape[0] == 0:
                if _bbox_crop_count == MAX_RETRY:
                    raise ValueError(f"It seems no way to use crop augmentation for this dataset. bbox: {bbox}, (i, j, h, w): {(i, j, h, w)}")
                bbox_candidate = self._crop_bbox(bbox, i, j, h, w)
                _bbox_crop_count += 1
            bbox = bbox_candidate
        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format((self.size_h, self.size_w))


class RandomResizedCrop(T.RandomResizedCrop):
    visualize = True

    def __init__(
        self,
        size: Union[int, List],
        scale: Union[float, List],
        ratio: Union[float, List],
        interpolation: str,
    ):
        interpolation = INVERSE_MODES_MAPPING[interpolation]
        # @illian01: antialias paramter always true (always use) for PIL image, and NetsPresso Trainer uses PIL image for augmentation.
        super().__init__(size, scale, ratio, interpolation)

    def _crop_bbox(self, bbox, i, j, h, w):
        area_original = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])

        bbox[..., 0:4:2] = np.clip(bbox[..., 0:4:2] - j, 0, w)
        bbox[..., 1:4:2] = np.clip(bbox[..., 1:4:2] - i, 0, h)

        area_cropped = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
        area_ratio = area_cropped / (area_original + 1)  # +1 for preventing ZeroDivisionError

        bbox = bbox[area_ratio >= BBOX_CROP_KEEP_THRESHOLD, ...]
        return bbox

    def forward(self, image, mask=None, bbox=None):
        w_orig, h_orig = image.size
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        if mask is not None:
            mask = F.resized_crop(mask, i, j, h, w, self.size, interpolation=T.InterpolationMode.NEAREST)
        if bbox is not None:
            # img = crop(img, top, left, height, width)
            bbox_candidate = self._crop_bbox(bbox, i, j, h, w)
            _bbox_crop_count = 1
            while bbox_candidate.shape[0] != 0:
                if _bbox_crop_count == MAX_RETRY:
                    raise ValueError(f"It seems no way to use crop augmentation for this dataset. bbox: {bbox}, (i, j, h, w): {(i, j, h, w)}")
                bbox_candidate = self._crop_bbox(bbox, i, j, h, w)
                _bbox_crop_count += 1
            bbox = bbox_candidate

            # img = resize(img, size, interpolation)
            w_cropped, h_cropped = np.clip(w_orig - j, 0, w), np.clip(h_orig - i, 0, h)
            target_w, target_h = (self.size, self.size) if isinstance(self.size, int) else self.size
            bbox[..., 0:4:2] *= float(target_w / w_cropped)
            bbox[..., 1:4:2] *= float(target_h / h_cropped)

        return image, mask, bbox

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomErasing(T.RandomErasing):
    visualize = True

    def __init__(
        self,
        p: float,
        scale: List,
        ratio: List,
        value: Optional[int],
        inplace: bool,
    ):
        scale = list(scale)
        ratio = list(ratio)
        super().__init__(p, scale, ratio, value, inplace)

    @staticmethod
    def get_params(
        img, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[int] = None
    ):
        img_w, img_h = img.size

        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = np.random.randint(255, size=(h, w)).astype('uint8')
                v = Image.fromarray(v).convert(img.mode)
            else:
                v = Image.new(img.mode, (w, h), value)

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, v

        # Return original image
        return 0, 0, img

    def forward(self, image, mask=None, bbox=None):
        if torch.rand(1) < self.p:
            x, y, v = self.get_params(image, scale=self.scale, ratio=self.ratio, value=self.value)
            image.paste(v, (y, x))
            # TODO: Object-aware
            return image, mask, bbox
        return image, mask, bbox


class TrivialAugmentWide(torch.nn.Module):
    """
    Based on the torchvision implementation.
    https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide
    """
    visualize = True

    def __init__(
        self,
        num_magnitude_bins: int,
        interpolation: str,
        fill: Optional[Union[List[int], int]],
    ) -> None:
        super().__init__()
        interpolation = INVERSE_MODES_MAPPING[interpolation]

        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, image, mask=None, bbox=None):
        fill = self.fill
        channels, height, width = F.get_dimensions(image)
        if isinstance(image, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        # TODO: Compute mask, bbox
        return _apply_op(image, op_name, magnitude, interpolation=self.interpolation, fill=fill), mask, bbox

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


class AutoAugment(T.AutoAugment):
    visualize = True

    def __init__(
        self,
        policy: str,
        interpolation: str,
        fill: Optional[List[float]],
    ) -> None:
        policy = T.AutoAugmentPolicy(policy)
        interpolation = INVERSE_MODES_MAPPING[interpolation]

        super().__init__(policy, interpolation, fill)

    def forward(self, image, mask=None, bbox=None):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(image)
        if isinstance(image, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(len(self.policies))

        op_meta = self._augmentation_space(10, (height, width))
        for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
            if probs[i] <= p:
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                image = _apply_op(image, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        # TODO: Compute mask, bbox
        return image, mask, bbox


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


class Normalize:
    visualize = False

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, bbox=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class ToTensor(T.ToTensor):
    visualize = False

    def __call__(self, image, mask=None, bbox=None):
        image = F.to_tensor(image)
        if mask is not None:
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        if bbox is not None:
            bbox = torch.as_tensor(np.array(bbox), dtype=torch.float)

        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + "()"
