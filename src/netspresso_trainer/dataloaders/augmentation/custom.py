import random
from collections.abc import Sequence
from typing import Dict, Optional

import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

BBOX_CROP_KEEP_THRESHOLD = 0.2
MAX_RETRY = 5

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

    def __init__(self, p):
        self.p = p

    def __call__(self, image, mask=None, bbox=None):
        w, _ = image.size
        if random.random() < self.p:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
            if bbox is not None:
                bbox[..., 0:4:2] = w - bbox[..., 0:4:2]
        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)


class RandomVerticalFlip:
    visualize = True

    def __init__(self, p):
        self.p = p

    def __call__(self, image, mask=None, bbox=None):
        _, h = image.size
        if random.random() < self.p:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)
            if bbox is not None:
                bbox[..., 1:4:2] = h - bbox[..., 1:4:2]
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

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.0):
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

    def __init__(self, size):

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