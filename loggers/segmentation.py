import numpy as np
import torch

from loggers.base import BaseCSVLogger, BaseImageSaver
from datasets.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


CSV_HEADER = ['epoch', 'train_loss', 'train_miou %', 'valid_miou %', 'valid_pixAcc %']

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = voc_color_map(n)
        self.cmap = torch.from_numpy(self.cmap[:n])
        
    def _convert(self, gray_image):
        assert len(gray_image.shape) == 2
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

    def __call__(self, gray_image):
        if len(gray_image.shape) == 3:
            images = []
            for _real_gray_image in gray_image:
                images.append(self._convert(_real_gray_image)[np.newaxis, ...])
            
            return np.concatenate(images, axis=0)
        elif len(gray_image.shape) == 2:
            return self._convert(gray_image)
        else:
            raise IndexError(f"gray_image.shape should be either 2 or 3, but {gray_image.shape} were indexed.")
        
        


def voc_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def magic_image_handler(img, num_example_image=1):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.transpose((1, 2, 0))
    elif img.ndim == 2:
        img = np.repeat(img[..., np.newaxis], 3, axis=2)
    elif img.ndim == 4:
        img = img[:min(img.shape[0], num_example_image)]  # first 4 batch
        img = np.concatenate(img, axis=-1)
        img = img.transpose((1, 2, 0))
    else:
        raise ValueError(f'img ndim is {img.ndim}, should be 2~4')

    min_, max_ = np.amin(img), np.amax(img)
    is_int_array = img.dtype in [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64]
    try_uint8 = (min_ >= 0 and max_ <= 255)

    if is_int_array and try_uint8:
        img = img.astype(np.uint8)
    else:
        if min_ >= 0 and max_ <= 1:
            img = (img * 255.0).astype(np.uint8)
        elif min_ >= -0.5 and max_ <= 0.5:
            img = ((img + 0.5) * 255.0).astype(np.uint8)
        elif min_ >= -1 and max_ <= 1:
            img = ((img + 1) / 2.0 * 255.0).astype(np.uint8)
        else:
            # denormalize with mean and std
            img = np.clip(img * (np.array(IMAGENET_DEFAULT_STD, dtype=np.float32) * 255.0) + np.array(IMAGENET_DEFAULT_MEAN, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)


    if img.shape[-1] != 1 and img.shape[-1] != 3:
        img = np.expand_dims(np.concatenate([img[..., i] for i in range(img.shape[-1])], axis=0), -1)
    img = np.clip(img, a_min=0, a_max=255)
    return img

class SegmentationCSVLogger(BaseCSVLogger):
    def __init__(self, csv_path):
        super(SegmentationCSVLogger, self).__init__(csv_path)
        self.header = CSV_HEADER
        self.update_header()


class SegmentationImageSaver(BaseImageSaver):
    def __init__(self, result_dir) -> None:
        super(BaseImageSaver, self).__init__(result_dir)
    
    def save_result(self, data):
        return