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
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_iou
from torchvision.transforms.autoaugment import _apply_op
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import _check_sequence_input

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

    def _get_transformed(self, image, label, mask, bbox, keypoint, visualize_for_debug, dataset):
        for t in self.transforms:
            if visualize_for_debug and not t.visualize:
                continue
            image, label, mask, bbox, keypoint = t(image=image, label=label, mask=mask, bbox=bbox, keypoint=keypoint, dataset=dataset)
        return image, label, mask, bbox, keypoint

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, visualize_for_debug=False, dataset=None, **kwargs):
        additional_targets_result = {k: None for k in kwargs if k in self.additional_targets}

        result_image, result_label, result_mask, result_bbox, result_keypoint = self._get_transformed(image=image, label=label, mask=mask, bbox=bbox, keypoint=keypoint, dataset=dataset, visualize_for_debug=visualize_for_debug)
        for key in additional_targets_result:
            if self.additional_targets[key] == 'mask':
                _, _, additional_targets_result[key], _, _ = self._get_transformed(image=image, label=label, mask=kwargs[key], bbox=None, keypoint=keypoint, dataset=dataset, visualize_for_debug=visualize_for_debug)
            elif self.additional_targets[key] == 'bbox':
                _, _, _, additional_targets_result[key], _ = self._get_transformed(image=image, label=label, mask=None, bbox=kwargs[key], keypoint=keypoint, dataset=dataset, visualize_for_debug=visualize_for_debug)
            else:
                del additional_targets_result[key]

        return_dict = {'image': result_image, 'label': result_label}
        if mask is not None:
            return_dict.update({'mask': result_mask})
        if bbox is not None:
            return_dict.update({'bbox': result_bbox})
        if keypoint is not None:
            return_dict.update({'keypoint': result_keypoint})
        return_dict.update(additional_targets_result)
        return return_dict

    def __repr__(self):
        compose_summary = "CustomCompose"
        compose_list = ",\n\t".join([str(t) for t in self.transforms])
        compose_summary += "(\n\t" + compose_list + "\n)"
        return compose_summary


class CenterCrop(T.CenterCrop):
    visualize = True

    def __init__(
        self,
        size: Union[int, List],
    ):
        super().__init__(size)

    def forward(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        # TODO: Compute mask, bbox, keypoint
        return F.center_crop(image, self.size), label, mask, bbox, keypoint

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class Identity:
    visualize = True

    def __init__(self):
        pass

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        return image, label, mask, bbox, keypoint

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Resize(T.Resize):
    visualize = True

    def __init__(
        self,
        size: Union[int, List],
        interpolation: str,
        max_size: Optional[int],
        resize_criteria: Optional[str],
    ):
        if isinstance(size, int):
            assert resize_criteria is not None, "If size is int, resize_criteria must be 'short' or 'long'"
        self.resize_criteria = resize_criteria
        interpolation = INVERSE_MODES_MAPPING[interpolation]
        # @illian01: antialias paramter always true (always use) for PIL image, and NetsPresso Trainer uses PIL image for augmentation.
        super().__init__(size, interpolation, max_size)

    def forward(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        w, h = image.size

        if isinstance(self.size, int) and self.resize_criteria == 'long':
            long_side, short_side = max(h, w), min(h, w)
            resize_factor = self.size / long_side
            target_size = [self.size, round(resize_factor * short_side)] if h > w else [round(resize_factor * short_side), self.size]
        else:
            target_size = self.size

        image = F.resize(image, target_size, self.interpolation, self.max_size, self.antialias)
        if mask is not None:
            mask = F.resize(mask, target_size, interpolation=T.InterpolationMode.NEAREST,
                            max_size=self.max_size)
        if bbox is not None:
            target_w, target_h = image.size # @illian01: Determine ratio according to the actual resized image
            bbox[..., 0:4:2] *= float(target_w / w)
            bbox[..., 1:4:2] *= float(target_h / h)
        # TODO: Compute keypoint
        return image, label, mask, bbox, keypoint

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, interpolation={1}, max_size={2}, antialias={3}, resize_criteria={4})".format(
            self.size, self.interpolation.value, self.max_size, self.antialias, self.resize_criteria)


class RandomHorizontalFlip:
    visualize = True

    def __init__(
        self,
        p: float,
    ):
        self.p: float = max(0., min(1., p))

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        w, _ = image.size
        if random.random() < self.p:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
            if bbox is not None:
                bbox[..., 2::-2] = w - bbox[..., 0:4:2]
            if keypoint is not None:
                keypoint = keypoint[:, dataset.flip_indices] # flip_indices must be defined in config (swap)
                keypoint[..., 0] = w - keypoint[..., 0]
        return image, label, mask, bbox, keypoint

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)


class RandomVerticalFlip:
    visualize = True

    def __init__(
        self,
        p: float,
    ):
        self.p: float = max(0., min(1., p))

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        _, h = image.size
        if random.random() < self.p:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)
            if bbox is not None:
                bbox[..., 3::-2] = h - bbox[..., 1:4:2]
            if keypoint is not None:
                keypoint = keypoint[:, dataset.flip_indices] # flip_indices must be defined in config (swap)
                keypoint[..., 1] = h - keypoint[..., 1]
        return image, label, mask, bbox, keypoint

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)


class Pad:
    visualize = True

    def __init__(
        self,
        size: Union[int, List],
        fill: Union[int, List],
    ):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) != 2:
            raise ValueError("If size is a sequence, it should have 2 values")
        self.new_h = size[0] if isinstance(size, Sequence) else size
        self.new_w = size[1] if isinstance(size, Sequence) else size
        self.fill = fill
        self.padding_mode = 'constant' # @illian: Fix as constant. I think other options are not gonna used well.

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        if not isinstance(image, (torch.Tensor, Image.Image)):
            raise TypeError("Image should be Tensor or PIL.Image. Got {}".format(type(image)))

        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            w, h = image.shape[-1], image.shape[-2]

        w_pad_needed = max(0, self.new_w - w)
        h_pad_needed = max(0, self.new_h - h)
        # @illian01: I think we don't need to pad in all direction.
        padding_ltrb = [0, 0, w_pad_needed, h_pad_needed]
        '''
        padding_ltrb = [w_pad_needed // 2,
                        h_pad_needed // 2,
                        w_pad_needed // 2 + w_pad_needed % 2,
                        h_pad_needed // 2 + h_pad_needed % 2]
        '''
        image = F.pad(image, padding_ltrb, fill=self.fill, padding_mode=self.padding_mode)
        if mask is not None:
            mask = F.pad(mask, padding_ltrb, fill=255, padding_mode=self.padding_mode)
        if bbox is not None:
            padding_left, padding_top, _, _ = padding_ltrb
            bbox[..., 0:4:2] += padding_left
            bbox[..., 1:4:2] += padding_top
        # TODO: Compute keypoint
        return image, label, mask, bbox, keypoint

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

    def forward(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
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

        return image, label, mask, bbox, keypoint

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
        fill: int,
    ):

        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size_h = size[0] if isinstance(size, Sequence) else size
        self.size_w = size[1] if isinstance(size, Sequence) else size
        self.image_pad_if_needed = Pad((self.size_h, self.size_w), fill=fill)

    def _crop_bbox(self, bbox, i, j, h, w):
        area_original = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])

        bbox[..., 0:4:2] = np.clip(bbox[..., 0:4:2] - j, 0, w)
        bbox[..., 1:4:2] = np.clip(bbox[..., 1:4:2] - i, 0, h)

        area_cropped = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
        area_ratio = area_cropped / (area_original + 1)  # +1 for preventing ZeroDivisionError

        bbox = bbox[area_ratio >= BBOX_CROP_KEEP_THRESHOLD, ...]
        return bbox

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        image, _, mask, bbox, _ = self.image_pad_if_needed(image=image, mask=mask, bbox=bbox)
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
        # TODO: Compute keypoint
        return image, label, mask, bbox, keypoint

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

    def forward(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
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

        return image, label, mask, bbox, keypoint

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

    def forward(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        if torch.rand(1) < self.p:
            x, y, v = self.get_params(image, scale=self.scale, ratio=self.ratio, value=self.value)
            image.paste(v, (y, x))
            # TODO: Object-aware
            return image, label, mask, bbox, keypoint
        return image, label, mask, bbox, keypoint


class RandomIoUCrop:
    """
    Based on the torchvision implementation.
    https://pytorch.org/vision/stable/_modules/torchvision/transforms/v2/_geometry.html#RandomIoUCrop
    """
    visualize = True
    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        min_aspect_ratio: float,
        max_aspect_ratio: float,
        p: float,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials
        self.p = p

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        if not isinstance(image, (torch.Tensor, Image.Image)):
            raise TypeError("Image should be Tensor or PIL.Image. Got {}".format(type(image)))

        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            w, h = image.shape[-1], image.shape[-2]
        if random.random() < self.p:
            while True:
                idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
                min_jaccard_overlap = self.options[idx]
                if min_jaccard_overlap >= 1.0:
                    return image, label, mask, bbox, keypoint
                for _ in range(self.trials):
                    r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                    new_w = int(w * r[0])
                    new_h = int(h * r[1])
                    aspect_ratio = new_w / new_h
                    if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                        continue

                    # check for 0 area crops
                    r = torch.rand(2)
                    left = int((w - new_w) * r[0])
                    top = int((h - new_h) * r[1])
                    right = left + new_w
                    bottom = top + new_h
                    if left == right or top == bottom:
                        continue
                    xyxy_bboxes = bbox
                    cx = 0.5 * (xyxy_bboxes[..., 0] + xyxy_bboxes[..., 2])
                    cy = 0.5 * (xyxy_bboxes[..., 1] + xyxy_bboxes[..., 3])
                    is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                    if not is_within_crop_area.any():
                        continue
                    xyxy_bboxes = torch.tensor(xyxy_bboxes[is_within_crop_area])
                    ious = box_iou(
                        xyxy_bboxes,
                        torch.tensor([[left, top, right, bottom]], dtype=xyxy_bboxes.dtype,
                                     device=xyxy_bboxes.device),)
                    if ious.max() < min_jaccard_overlap:
                        continue
                    image = F.crop(image, top, left, new_h, new_w)
                    if bbox is not None:
                        bbox = self._crop_bbox(bbox, top, left, new_h, new_w)
                    return image, label, mask, bbox, keypoint

        return image, label, mask, bbox, keypoint

    def _crop_bbox(self, bbox, i, j, h, w):
        bbox[..., 0:4:2] = np.clip(bbox[..., 0:4:2] - j, 0, w)
        bbox[..., 1:4:2] = np.clip(bbox[..., 1:4:2] - i, 0, h)
        return bbox

class RandomZoomOut:
    """
    Based on the torchvision implementation.
    https://pytorch.org/vision/0.18/_modules/torchvision/transforms/v2/_geometry.html#RandomZoomOut
    """
    visualize = True

    def __init__(
        self,
        fill: int,
        side_range: List[float],
        p: float,

    ) -> None:
        _check_sequence_input(side_range, "side_range", req_sizes=(2,))
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError("Invalid side range provided {}.".format(side_range))
        self.fill = fill
        self.padding_mode = 'constant'
        self.side_range = side_range
        self.p = p

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        if not isinstance(image, (torch.Tensor, Image.Image)):
            raise TypeError("Image should be Tensor or PIL.Image. Got {}".format(type(image)))

        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            w, h = image.shape[-1], image.shape[-2]

        if random.random() < self.p:
            r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
            canvas_width = int(w * r)
            canvas_height = int(h * r)

            r = torch.rand(2)
            left= int((canvas_width - w) * r[0])
            top = int((canvas_height - h) * r[1])
            right = canvas_width - (left + w)
            bottom = canvas_height - (top + h)
            padding_ltrb = [left, top, right, bottom]
            image = F.pad(image, padding_ltrb, fill=self.fill, padding_mode=self.padding_mode)
            if mask is not None:
                mask = F.pad(mask, padding_ltrb, fill=255, padding_mode=self.padding_mode)
            if bbox is not None:
                padding_left, padding_top, _, _ = padding_ltrb
                bbox[..., 0:4:2] += padding_left
                bbox[..., 1:4:2] += padding_top
            #TODO: Compute Keypoint

        return image, label, mask, bbox, keypoint


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

    def forward(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
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
        return _apply_op(image, op_name, magnitude, interpolation=self.interpolation, fill=fill), label, mask, bbox, keypoint

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

    def forward(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
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
        return image, label, mask, bbox, keypoint


class HSVJitter:
    visualize = True

    def __init__(
        self,
        h_mag: int,
        s_mag: int,
        v_mag: int,
    ):
        self.h_mag = h_mag
        self.s_mag = s_mag
        self.v_mag = v_mag

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        hsv_augs = np.random.uniform(-1, 1, 3) * [self.h_mag, self.s_mag, self.v_mag]  # random gains
        hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
        hsv_augs = hsv_augs.astype(np.int16)

        img_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

        image = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        image = Image.fromarray(image)
        return image, label, mask, bbox, keypoint

    def __repr__(self):
        return self.__class__.__name__ + "(h_mag={0}, s_mag={1}, v_mag={2})".format(
            self.h_mag, self.s_mag, self.v_mag)


class RandomResize:
    visualize = True

    def __init__(
        self,
        base_size: List,
        stride: int,
        random_range: int,
        interpolation: str,
    ):
        self.base_size = base_size
        self.stride = stride
        self.random_range = random_range
        self.resize = Resize(size=base_size, interpolation=interpolation, max_size=None, resize_criteria=None)
        self.counter = 0

    def random_set(self):
        delta = self.stride * random.randint(-self.random_range, self.random_range)
        size = [self.base_size[0] + delta, self.base_size[1] + delta]
        self.resize.size = size

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        """
        @illian01:
            Count random resized samples.
            If one batch completed, randomly reset target size and set counter to 0.
        """
        if self.counter == dataset.batch_size:
            self.random_set()
            self.counter = 0
        image, label, mask, bbox, keypoint = self.resize(image, label, mask, bbox, dataset)
        self.counter += 1
        return image, label, mask, bbox, keypoint

    def __repr__(self):
        return self.__class__.__name__ + "(base_size={0}, stride={1}, random_range={2})".format(
            self.base_size, self.stride, self.random_range
        )


class PoseTopDownAffine:
    """
    Based on the mmpose implementation.
    https://github.com/open-mmlab/mmpose
    """
    visualize = False

    def __init__(
        self,
        scale: List,
        scale_prob: float,
        translate: float,
        translate_prob: float,
        rotation: int,
        rotation_prob: float,
        size: List,
    ):
        self.scale = scale
        self.scale_prob = scale_prob
        self.translate = translate
        self.translate_prob = translate_prob
        self.rotation = rotation
        self.rotation_prob = rotation_prob
        self.size = size

        assert len(self.size) == 2, "Target size must be a list of length 2"

    def trunc_normal_(self, low: float, high: float, size: Tuple = (1)):
        """
        Copied from torch.nn.init._no_grad_trunc_normal_
        This is instead of scipy.stats.tuncnorm (Not to add dependency)
        """
        mean = 0.
        std = 1.
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        with torch.no_grad():
            tensor = torch.zeros(size).to(torch.float32)

            lower = norm_cdf((low - mean) / std)
            upper = norm_cdf((high - mean) / std)

            tensor.uniform_(2 * lower - 1, 2 * upper - 1)

            tensor.erfinv_()

            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)

            tensor.clamp_(min=low, max=high)
            return tensor.numpy()

    def _rotate_point(self, pt: np.ndarray, angle_rad: float):
        """
        Rotate a point by an angle.
        """
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        rot_mat = np.array([[cs, -sn], [sn, cs]])
        return rot_mat @ pt

    def _get_3rd_point(self, a: np.ndarray, b: np.ndarray):
        """
        To calculate the affine matrix, three pairs of points are required. This
        function is used to get the 3rd point, given 2D points a & b.
        """
        direction = a - b
        c = b + np.r_[-direction[1], direction[0]]
        return c

    def get_warp_matrix(self, box_center: np.ndarray, box_wh: np.ndarray, rot: float):
        """
        Calculate the affine transformation matrix that can warp the bbox area
        in the input image to the output size.
        """
        shift = (0, 0) # Fix as 0
        assert len(box_center) == 2
        assert len(box_wh) == 2
        assert len(shift) == 2

        shift = np.array(shift)
        src_w, src_h = box_wh[:2]
        dst_w, dst_h = self.size

        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(np.array([src_w * -0.5, 0.]), rot_rad)
        dst_dir = np.array([dst_w * -0.5, 0.])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = box_center + box_wh * shift
        src[1, :] = box_center + src_dir + box_wh * shift

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return warp_mat

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        image = np.array(image)

        # This is only for one instance (bbox)
        bbox_ = bbox[0].reshape(2, 2).copy()
        bbox_center = bbox_.sum(axis=0) / 2.
        bbox_wh = bbox_[1] - bbox_[0]
        bbox_ = bbox_.reshape(-1)

        # Randomly scale, shift box, and get random rotation degree which is applied in affine transform
        scale_min, scale_max = self.scale
        mu = (scale_max + scale_min) * 0.5
        sigma = (scale_max - scale_min) * 0.5
        scale = self.trunc_normal_(low=-1., high=1., size=(1)) * sigma + mu
        scale = np.where(np.random.rand(1) < self.scale_prob, scale, 1.)

        translate = self.trunc_normal_(low=-1., high=1., size=(2)) * self.translate
        translate = np.where(np.random.rand(1) < self.translate_prob, translate, 0.)

        bbox_center = bbox_center + bbox_wh * translate
        bbox_wh = bbox_wh * scale
        rot = (self.rotation * self.trunc_normal_(low=-1., high=1., size=(1))).item()

        # Get warping matrix
        warp_mat = self.get_warp_matrix(bbox_center, bbox_wh, rot)

        # Apply affine transform
        image = cv2.warpAffine(image, warp_mat, self.size, flags=cv2.INTER_LINEAR)
        image = Image.fromarray(image) # return as PIL

        # Compute keypoint. Note that this is only for one instance.
        # ``keypoint.shape`` should be (1, num_keypoints, 3)
        keypoint[..., :2] = cv2.transform(keypoint[..., :2], warp_mat)

        # Now, bbox is same with image size
        bbox_ = np.array([0, 0] + self.size[::-1]).astype('float32')
        bbox[0] = bbox_

        return image, label, mask, bbox, keypoint

    def __repr__(self):
        return self.__class__.__name__ + f"(scale={self.scale}, translate={self.translate}, rotation={self.rotation}, size={self.size})"

class Normalize:
    visualize = False

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, label, mask, bbox, keypoint

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class ToTensor:
    visualize = False

    def __init__(self, pixel_range):
        self.pixel_range = pixel_range

    def __call__(self, image, label=None, mask=None, bbox=None, keypoint=None, dataset=None):
        image = F.to_tensor(image) * self.pixel_range
        if mask is not None:
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        if bbox is not None:
            bbox = torch.as_tensor(np.array(bbox), dtype=torch.float)

        return image, label, mask, bbox, keypoint

    def __repr__(self):
        return self.__class__.__name__ + "()"
