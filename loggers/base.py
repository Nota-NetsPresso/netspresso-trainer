from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Union

import numpy as np
import PIL.Image as Image

CSV_FILENAME = "results.csv"
class BaseCSVLogger(ABC):
    def __init__(self, model, result_dir):
        super(BaseCSVLogger, self).__init__()
        self.model = model
        self.csv_path = Path(result_dir) / CSV_FILENAME
        self.header: List = []
        self.key_map: Dict = {}
        
        self._temp_row_dict = dict()
        
        if self.csv_path.exists():
            self.csv_path.unlink()
            
        self._epoch = None
    
    def init_epoch(self):
        self._epoch = 0
        
    @property
    def epoch(self):
        return self._epoch
    
    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch = int(value)
        
    def update_header(self, header: List):
        assert len(header) != 0
        self.header = header
        
        with open(self.csv_path, 'a') as f:
            f.write(",".join(self.header))
            f.write("\n")

    def _clear_temp(self):
        self._temp_row_dict = dict()
    
    def _update_with_list(self, data: List):
        if data is not None and len(data) != 0:
            with open(self.csv_path, 'a') as f:
                f.write(",".join([f"{x:.09f}" for x in data]))
                f.write("\n")
        self._clear_temp()
        return
    
    def _update_specific(self, data: Dict):
        for _key, _value in data.items():
            if _key not in self.header:
                raise AssertionError(f"The given key ({_key}) is not in {self.header}!")
            if _key not in self._temp_row_dict:
                self._temp_row_dict[_key] = _value
        
        if set(self.header) == set(self._temp_row_dict.keys()):
            self._update_with_list([self._temp_row_dict[_col] for _col in self.header])
        return
            
    def update(self, data=None, **kwargs):
        if isinstance(data, List):
            return self._update_with_list(data)
        if isinstance(data, Dict):
            return self._update_specific(data)
        # if isinstance(data, type(None)):
        #     return self._update_specific(kwargs)
        
        raise AssertionError(f"Type of data should be either List or Dict! Current: {type(data)}")
    
    def _convert_as_csv_record(self, scalar_dict: Dict, prefix='train'):
        converted_dict = {}
        for k, v in scalar_dict.items():
            if f"{prefix}/{k}" not in self.key_map:
                continue
            record_key = self.key_map[f"{prefix}/{k}"]
            assert record_key in self.header, f"{record_key} not in {self.header}"
            
            converted_dict.update({record_key: v})
        return converted_dict
    
    def __call__(self, train_losses, train_metrics, valid_losses=None, valid_metrics=None):
        assert len(self.header) != 0
        assert len(self.key_map) != 0
        
        csv_record_dict = {'epoch': self._epoch}
        converted_train_losses = self._convert_as_csv_record(train_losses, prefix='train')
        converted_train_metrics = self._convert_as_csv_record(train_metrics, prefix='train')
        csv_record_dict.update(converted_train_losses)
        csv_record_dict.update(converted_train_metrics)
        
        if valid_losses is not None:
            converted_valid_losses = self._convert_as_csv_record(valid_losses, prefix='valid')
            csv_record_dict.update(converted_valid_losses)
        if valid_metrics is not None:
            converted_valid_metrics = self._convert_as_csv_record(valid_metrics, prefix='valid')
            csv_record_dict.update(converted_valid_metrics)
        
        self.update(csv_record_dict)

    
class BaseImageSaver(ABC):
    def __init__(self, model, result_dir) -> None:
        super(BaseImageSaver, self).__init__()
        self.model = model
        self.result_dir = Path(result_dir)
        self._epoch = None
    
    def init_epoch(self):
        self._epoch = 0
        
    @property
    def epoch(self):
        return self._epoch
    
    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch = int(value)
        
    @staticmethod
    def magic_visualizer(image: Union[np.ndarray, Image.Image, str, Path], size: Optional[Tuple]=None) -> np.ndarray:
        pass
    
    @abstractmethod
    def save_result(self, data):
        raise NotImplementedError

    def __call__(self, train_images, valid_images):
        pass

