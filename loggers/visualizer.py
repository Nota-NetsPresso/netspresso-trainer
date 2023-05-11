import numpy as np
import torch

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = _voc_color_map(n)
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