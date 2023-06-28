import random
from typing import Sequence, Optional, Dict
from collections import Sequence

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import PIL.Image as Image

BBOX_CROP_KEEP_THRESHOLD = 0.2
MAX_RETRY = 5
class Compose:
    def __init__(self, transforms, additional_targets: Dict={}):
        self.transforms = transforms
        self.additional_targets = additional_targets
    
    def _get_transformed(self, image, mask, bbox):
        for t in self.transforms:
            image, mask, bbox = t(image=image, mask=mask, bbox=bbox)
        return image, mask, bbox

    def __call__(self, image, mask=None, bbox=None, **kwargs):
        additional_targets_result = {k: None for k in kwargs.keys() if k in self.additional_targets}
        
        result_image, result_mask, result_bbox = self._get_transformed(image=image, mask=mask, bbox=bbox)
        for key in additional_targets_result.keys():
            if self.additional_targets[key] == 'mask':
                _, additional_targets_result[key], _ = self._get_transformed(image=image, mask=kwargs[key], bbox=None)
            elif self.additional_targets[key] == 'bbox':
                _, _, additional_targets_result[key] = self._get_transformed(image=image, mask=None, bbox=kwargs[key])
            else:
                del additional_targets_result[key]

        return_dict = {'image': result_image}
        if mask is not None:
            return_dict.update({'mask': result_mask})
        if bbox is not None:
            return_dict.update({'bbox': result_bbox})
        return_dict.update(additional_targets_result)
        return return_dict
    
class Identity:
    def __init__(self):
        pass

    def __call__(self, image, mask=None, bbox=None):
        return image, mask, bbox

class Pad(T.Pad):
    def forward(self, image, mask=None, bbox=None):
        image = F.pad(image, self.padding, self.fill, self.padding_mode)
        if mask is not None:
            mask = F.pad(mask, self.padding, fill=255, padding_mode=self.padding_mode)
        if bbox is not None:
            if not isinstance(self.padding, Sequence):
                target_padding = [self.padding]
            else:
                target_padding = self.padding
                
            padding_left, padding_top, _, _ = \
                target_padding * (4 / len(target_padding)) # supports 1, 2, 4 length
                
            bbox[..., 0:4:2] += padding_left
            bbox[..., 1:4:2] += padding_top
            
        return image, mask, bbox

class Resize(T.Resize):
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

class RandomHorizontalFlip:
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

class RandomVerticalFlip:
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

class PadIfNeeded:
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

class ColorJitter(T.ColorJitter):
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

class RandomCrop:
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
    
class RandomResizedCrop(T.RandomResizedCrop):
    
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

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, bbox=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask, bbox

class ToTensor(T.ToTensor):
    def __call__(self, image, mask=None, bbox=None):
        image = F.to_tensor(image)
        if mask is not None:
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        if bbox is not None:
            bbox = torch.as_tensor(np.array(bbox), dtype=torch.float)

        return image, mask, bbox


if __name__ == '__main__':
    from pathlib import Path
    
    import PIL.Image as Image
    import albumentations as A
    import cv2
    import numpy as np
    input_filename = Path("astronaut.jpg")
    im = Image.open(input_filename)
    im_array = np.array(im)
    print(f"Original image size (in array): {im_array.shape}")
    
    """Pad"""
    torch_aug = PadIfNeeded(size=(1024, 1024))
    im_torch_aug, _, _ = torch_aug(im)
    im_torch_aug.save(f"{input_filename.stem}_torch{input_filename.suffix}")
    print(f"Aug image size (from torchvision): {np.array(im_torch_aug).shape}")
    
    album_aug = A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT)
    im_album_aug: np.ndarray = album_aug(image=im_array)['image']
    Image.fromarray(im_album_aug).save(f"{input_filename.stem}_album{input_filename.suffix}")
    print(f"Aug image size (from albumentations): {im_album_aug.shape}")

    print(np.all(np.array(im_torch_aug) == im_album_aug), np.mean(np.abs(np.array(im_torch_aug) - im_album_aug)))
    
