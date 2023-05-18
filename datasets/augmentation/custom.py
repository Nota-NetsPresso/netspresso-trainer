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

    def forward(self, img):
        if not isinstance(img, (torch.Tensor, Image.Image)):
            raise TypeError("Image should be Tensor or PIL.Image. Got {}".format(type(img)))
        
        if isinstance(img, Image.Image):
            w, h = img.size
        else:
            w, h = img.shape[-1], img.shape[-2]
        
        w_pad_needed = max(0, self.new_w - w)
        h_pad_needed = max(0, self.new_h - h)
        padding_ltrb = [w_pad_needed // 2,
                        h_pad_needed // 2,
                        w_pad_needed // 2 + w_pad_needed % 2,
                        h_pad_needed // 2 + h_pad_needed % 2]
        return F.pad(img, padding_ltrb, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(min_size={0}, fill={1}, padding_mode={2})'.\
            format((self.new_h, self.new_w), self.fill, self.padding_mode)


if __name__ == '__main__':
    import PIL.Image as Image
    im = Image.open("astronaut.jpg")
    