from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import PIL.Image as Image
    

class BaseCSVLogger(ABC):
    def __init__(self, csv_path):
        super(BaseCSVLogger, self).__init__()
        self.csv_path = Path(csv_path)
        self.header = []
        
        self._temp_row_dict = dict()
        
        if self.csv_path.exists():
            self.csv_path.unlink()
        
    def update_header(self):
        assert len(self.header) != 0
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
    
    def _update_specific(self, data: Dict):
        for _key, _value in data.items():
            if _key not in self.header:
                raise AssertionError(f"The given key ({_key}) is not in {self.header}!")
            if _key not in self._temp_row_dict:
                self._temp_row_dict[_key] = _value
        
        if set(self.header) == set(self._temp_row_dict.keys()):
            self._update_with_list([self._temp_row_dict[_col] for _col in self.header])
            
    def update(self, data=None, **kwargs):
        if isinstance(data, List):
            return self._update_with_list(data)
        if isinstance(data, Dict):
            return self._update_specific(data)
        # if isinstance(data, type(None)):
        #     return self._update_specific(kwargs)
        
        raise AssertionError(f"Type of data should be either List or Dict! Current: {type(data)}")

class BaseVisualizer(ABC):
    def __init__(self, result_dir) -> None:
        super(BaseVisualizer, self).__init__()
        self.result_dir = Path(result_dir)
        
    @staticmethod
    def magic_visualizer(image: Union[np.ndarray, Image.Image, str, Path], size: Optional[Tuple]=None) -> np.ndarray:
        pass
    
    @abstractmethod
    def save_result(self, data):
        raise NotImplementedError

class InferenceReporter(ABC):
    def __init__(self, csv_logger: Optional[BaseCSVLogger], visualizer: Optional[BaseVisualizer]) -> None:
        super(InferenceReporter, self).__init__()
        self.csv_logger = csv_logger
        self.visualizer = visualizer
        
    def update(self, data):
        assert isinstance(self.csv_logger, BaseCSVLogger), "`csv_logger` is not initialized!"
        self.csv_logger.update(data)
    
    def log_prediction(self, data, epoch: int):
        assert isinstance(self.visualizer, BaseVisualizer), "`visualizer` is not initialized!"
        self.visualizer.save_result(data)
