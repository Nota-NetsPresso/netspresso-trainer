from typing import Sequence

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

class Pad(T.Pad):
    pass

class Resize(T.Resize):
    pass

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    pass



if __name__ == '__main__':
    import PIL.Image as Image
    im = Image.open("astronaut.jpg")
    