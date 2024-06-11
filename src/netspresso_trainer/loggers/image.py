# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

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

        for k, v_list in image_dict.items():
            for idx, v in enumerate(v_list):
                assert isinstance(v, np.ndarray)
                if epoch is None:
                    self.save_ndarray_as_image(v, f"{prefix_dir}/{idx:03d}_{k}.png", dataformats='HWC')
                else:
                    self.save_ndarray_as_image(v, f"{prefix_dir}/{epoch:04d}_{idx:03d}_{k}.png", dataformats='HWC')

    def __call__(
        self,
        prefix: Literal['training', 'validation', 'evaluation', 'inference'],
        epoch: Optional[int] = None,
        images: Optional[List] = None,
        **kwargs
    ):
        if images is not None:
            self.save_result(images, prefix=prefix, epoch=epoch)
