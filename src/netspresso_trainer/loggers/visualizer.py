from typing import List, Tuple

import cv2
import numpy as np
import torch

from ..dataloaders import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _voc_color_map(N=256, normalized=False):
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

class DetectionVisualizer:
    def __init__(self, class_map, pallete=None):
        n = len(class_map)
        if pallete is None:
            self.cmap = _voc_color_map(n)
        else:
            self.cmap = np.array(pallete[:n], dtype=np.uint8)
        self.class_map = class_map


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
        mask = (gray_image == 255)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

    def __call__(self, results: List[Tuple[np.ndarray, np.ndarray]], images=None):

        return_images = []
        for image, result in zip(images, results):
            image = image.copy()
            bbox_list, class_index_list = result
            for bbox_label, class_label in zip(bbox_list, class_index_list):
                class_name = self.class_map[class_label]

                # unnormalize depending on the visualizing image size
                x1 = int(bbox_label[0])
                y1 = int(bbox_label[1])
                x2 = int(bbox_label[2])
                y2 = int(bbox_label[3])
                color = self.cmap[class_label].tolist()

                image = cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
                text_size, _ = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_w, text_h = text_size
                image = cv2.rectangle(image, (x1, y1-5-text_h), (x1+text_w, y1), color=color, thickness=-1)
                image = cv2.putText(image, str(class_name), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return_images.append(image[np.newaxis, ...])
        return_images = np.concatenate(return_images, axis=0)
        return return_images


class SegmentationVisualizer:
    def __init__(self, class_map, pallete=None):
        n = len(class_map)
        if pallete is None:
            self.cmap = _voc_color_map(n)
        else:
            self.cmap = np.array(pallete[:n], dtype=np.uint8)
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
        mask = (gray_image == 255)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

    def __call__(self, results, images=None):
        if len(results.shape) == 3:
            result_images = []
            for _real_gray_image in results:
                result_images.append(self._convert(_real_gray_image)[np.newaxis, ...])

            return np.concatenate(result_images, axis=0)
        elif len(results.shape) == 2:
            return self._convert(results)
        else:
            raise IndexError(f"gray_image.shape should be either 2 or 3, but {results.shape} were indexed.")


def _as_image_array(img: np.ndarray):
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


def magic_image_handler(img):
    if img.ndim == 3:
        img = img.transpose((1, 2, 0)) if img.shape[0] == 3 else img  # H x W x C
        return _as_image_array(img)
    elif img.ndim == 2:
        img = np.repeat(img[..., np.newaxis], 3, axis=2)
        return _as_image_array(img)
    elif img.ndim == 4:
        img_new = []
        for _img in img:
            _img = _img.transpose((1, 2, 0)) if _img.shape[0] == 3 else _img  # H x W x C
            img_new.append(_as_image_array(_img))
        return np.array(img_new)
    else:
        raise ValueError(f'img ndim is {img.ndim}, should be either 2, 3, or 4')
