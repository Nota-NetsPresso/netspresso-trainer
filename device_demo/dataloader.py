import glob
from pathlib import Path
from itertools import chain

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg',
                  '.bmp', '.tif', '.tiff', '.dng', '.webp', '.mpo')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def resize_img(img, size):
    w, h = img.size
    long_side, short_side = max(h, w), min(h, w)
    resize_factor = size / long_side
    target_size = [size, round(resize_factor * short_side)] if h > w else [round(resize_factor * short_side), size]
    img = F.resize(img, target_size, InterpolationMode.BILINEAR, None, True)
    return img

def pad_img(img, size):
    w, h = img.size
    w_pad_needed = max(0, size - w)
    h_pad_needed = max(0, size - h)
    padding_ltrb = [0, 0, w_pad_needed, h_pad_needed]
    img = F.pad(img, padding_ltrb, fill=114, padding_mode='constant')
    return img


def preprocess(img, size):
    img = resize_img(img, size)
    img = pad_img(img, size)

    img = np.array(img)
    img = img.astype('float') / 255.0
    img = img - np.array(IMAGENET_DEFAULT_MEAN).reshape(1, 1, -1)
    img = img / np.array(IMAGENET_DEFAULT_STD).reshape(1, 1, -1)
    img = img[np.newaxis, ...]
    return img


class LoadDirectory:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.images = []
        for ext in IMG_EXTENSIONS:
            for file in chain(self.path.glob(f'*{ext}'), self.path.glob(f'*{ext.upper()}')):
                self.images.append(file)

    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == len(self.images):
            raise StopIteration

        img_path = self.images[self.count]
        original_img = Image.open(img_path)
        self.count += 1

        return original_img

    def __len__(self):
        len(self.images)


class LoadCamera:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)

    def __iter__(self):
        return self

    def __next__(self):
        success, img = self.cap.read()
        if success:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        else:
            raise IOError("Failed to read camera frame")

    def __len__(self):
        return 1e12
