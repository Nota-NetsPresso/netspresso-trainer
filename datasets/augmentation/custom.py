import random
from typing import Sequence

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None, bbox=None, **kwargs):
        for t in self.transforms:
            image, mask, bbox = t(image=image, mask=mask, bbox=bbox, **kwargs)
        return image, mask, bbox

class Pad(T.Pad):
    def forward(self, image, mask=None, bbox=None):
        image = F.pad(image, self.padding, self.fill, self.padding_mode)
        if mask is not None:
            mask = F.pad(mask, self.padding, fill=255, padding_mode=self.padding_mode)
        return image, mask, bbox

class Resize(T.Resize):
    def forward(self, image, mask=None, bbox=None):
        image = F.resize(image, self.size, self.interpolation, self.max_size, self.antialias)
        if mask is not None:
            mask = F.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST,
                            max_size=self.max_size, antialias=False)
        return image, mask, bbox

class RandomHorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, image, mask=None, bbox=None):
        if random.random() < self.p:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask, bbox

class RandomVerticalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, image, mask=None, bbox=None):
        if random.random() < self.p:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)
        return image, mask, bbox

class PadIfNeeded(torch.nn.Module):
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

    def forward(self, image, mask=None, bbox=None):
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
        return image, mask, bbox

    def __repr__(self):
        return self.__class__.__name__ + '(min_size={0}, fill={1}, padding_mode={2})'.\
            format((self.new_h, self.new_w), self.fill, self.padding_mode)

class ColorJitter(T.ColorJitter):
    def forward(self, image, mask=None, bbox=None):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

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
        self.size = size
        self.image_pad_if_needed = PadIfNeeded(self.size)
        self.mask_pad_if_needed = PadIfNeeded(self.size, fill=255)

    def __call__(self, image, mask=None, bbox=None):
        image = self.image_pad_if_needed(image)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        if mask is not None:
            mask = self.mask_pad_if_needed(mask)
            mask = F.crop(mask, *crop_params)
        return image, mask, bbox
    
class RandomResizedCrop(T.RandomResizedCrop):
    def forward(self, image, mask=None, bbox=None):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        if mask is not None:
            mask = F.resized_crop(mask, i, j, h, w, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, mask, bbox

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, bbox=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
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
    
