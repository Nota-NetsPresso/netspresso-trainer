from pathlib import Path
import re
from dataloaders.utils.parsers.misc import natural_key

from dataloaders.utils.parsers.parser import Parser
from dataloaders.utils.parsers.class_map import load_class_map
from dataloaders.utils.constants import IMG_EXTENSIONS



def _natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_custom_images_and_targets(root, dir_to_idx, types=IMG_EXTENSIONS, sort=True):
    images_and_targets = []
    for dir_name, dir_idx in dir_to_idx.items():
        _dir = Path(root) / dir_name
        for ext in types:
            images_and_targets.extend([(str(file), dir_idx) for file in _dir.glob(f'*{ext}')])
            images_and_targets.extend([(str(file), dir_idx) for file in _dir.glob(f'*{ext.upper()}')])

    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: _natural_key(k[0]))

    return images_and_targets


def load_custom_class_map(root, map_or_filename=None):

    dir_list = [x.name for x in Path(root).iterdir() if x.is_dir()]

    if map_or_filename is None:  # class_name == dir_name
        dir_to_idx = {v.strip(): k for k, v in enumerate(dir_list)}
        idx_to_class = {k: v.strip() for k, v in enumerate(dir_list)}

    else:
        if isinstance(map_or_filename, dict):
            assert dict, "class_map dict must be non-empty"
            dir_to_idx = {v.strip(): k for k, v in enumerate(map_or_filename.keys())}
            idx_to_class = {k: v.strip() for k, v in enumerate(map_or_filename.values())}
            return dir_to_idx, idx_to_class

        class_map_path = Path(map_or_filename)

        if not class_map_path.exists():
            class_map_path = Path(root) / class_map_path
            assert class_map_path.exists(), f"Cannot locate specified class map file {map_or_filename}!"

        class_map_ext = class_map_path.suffix.lower()
        assert class_map_ext == '.txt', f"Unsupported class map file extension ({class_map_ext})!"

        with open(class_map_path) as f:
            map_data = [x.strip().split(' ') for x in f.readlines()]
            dir_list_from_map = [x[0] for x in map_data]
            assert set(dir_list).issubset(set(dir_list_from_map)), \
                f"Found unknown directory in ({root}) whose class is not defined: {set(dir_list).difference(set(dir_list_from_map))}"
            class_list_from_map = [' '.join(x[1:]) for x in map_data]
            dir_to_idx = {v.strip(): k for k, v in enumerate(dir_list_from_map)}
            idx_to_class = {k: v.strip() for k, v in enumerate(class_list_from_map)}

    return dir_to_idx, idx_to_class


class ParserImageFolder(Parser):

    def __init__(self, root, class_map=None):
        super().__init__()

        self.root = root
        self._dir_to_idx, self._idx_to_class = load_custom_class_map(root, class_map)
        self.samples = find_custom_images_and_targets(root, dir_to_idx=self._dir_to_idx, types=IMG_EXTENSIONS)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

        self._num_classes = len(self._dir_to_idx)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False) -> str:
        filename = Path(self.samples[index][0])
        if basename:
            filename = filename.name
        elif not absolute:
            filename = filename.relative_to(self.root)
        return str(filename)

    @property
    def dir_to_idx(self):
        return self._dir_to_idx

    @property
    def idx_to_class(self):
        return self._idx_to_class

    @property
    def num_classes(self):
        assert self._num_classes is not None
        return self._num_classes
