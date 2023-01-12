from pathlib import Path
from typing import List, Dict

class BaseCSVLogger:
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
            self._update_all(data)
        elif isinstance(data, Dict):
            self._update_specific(data)
        elif isinstance(data, type(None)):
            self._update_specific(kwargs)
        else:
            raise AssertionError(f"Type of data should be either List or Dict! Current: {type(data)}")
        