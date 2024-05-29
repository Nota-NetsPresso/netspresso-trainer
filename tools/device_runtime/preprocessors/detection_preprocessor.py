import cv2
import numpy as np

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def resize_img(img, size):
    h, w = img.shape[:2]
    long_side, short_side = max(h, w), min(h, w)
    resize_factor = size / long_side
    target_size = [size, round(resize_factor * short_side)] if h < w else [round(resize_factor * short_side), size]
    img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_LINEAR)
    return img

def pad_img(img, size):
    h, w = img.shape[:2]
    padded = np.full((size, size, 3), 114, dtype='uint8')
    padded[:h, :w] = img
    return padded


class DetectionPreprocessor:

    def __init__(self, preprocess_conf):
        self.size = preprocess_conf.size

    def __call__(self, img):
        img = resize_img(img, self.size)
        img = pad_img(img, self.size)

        img = img.astype('float32') / 255.0
        img = img - np.array(IMAGENET_DEFAULT_MEAN).reshape(1, 1, -1).astype('float32')
        img = img / np.array(IMAGENET_DEFAULT_STD).reshape(1, 1, -1).astype('float32')
        img = img[np.newaxis, ...]
        return img