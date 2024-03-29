from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image as Image


class ImageSaver:
    def __init__(self, model, result_dir) -> None:
        super(ImageSaver, self).__init__()
        self.model = model
        self.save_dir: Path = Path(result_dir) / "result_image"
        self.save_dir.mkdir(exist_ok=True)

    def save_ndarray_as_image(self, image_array: np.ndarray, filename: Union[str, Path], dataformats: Literal['HWC', 'CHW'] = 'HWC'):
        assert image_array.ndim == 3
        if dataformats != 'HWC' and dataformats == 'CHW':
            image_array = image_array.transpose((1, 2, 0))

        # HWC
        assert image_array.shape[-1] in [1, 3]
        Image.fromarray(image_array.astype(np.uint8)).save(filename)
        return True

    def save_result(self, image_dict: Dict, prefix, epoch):
        prefix_dir: Path = self.save_dir / prefix
        prefix_dir.mkdir(exist_ok=True)

        for k, v in image_dict.items():
            assert isinstance(v, np.ndarray)
            assert v.ndim in [3, 4], \
                f"Array for saving as image should have dim of 3 or 4! Current: {v.ndim}"
            if v.ndim == 3:
                self.save_ndarray_as_image(v, f"{prefix_dir}/{epoch:04d}_{k}.png", dataformats='HWC')
            for idx, image in enumerate(v):
                if epoch is None:
                    filename = f"{prefix_dir}/{idx:03d}_{k}.png"
                else:
                    filename = f"{prefix_dir}/{epoch:04d}_{idx:03d}_{k}.png"
                self.save_ndarray_as_image(image, filename, dataformats='HWC') # TODO: get dataformats option from outside

    def __call__(
        self,
        prefix: Literal['training', 'validation', 'evaluation', 'inference'],
        epoch: Optional[int] = None,
        images: Optional[List] = None,
        **kwargs
    ):
        if images is not None:
            self.save_result(images, prefix=prefix, epoch=epoch)
