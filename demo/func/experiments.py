from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

OUTPUT_DIR = Path("./outputs")

COLUMN_NAME_AS = {
    "is_fx_retrain": "🦴",
    "id": "Name",
    "model": "Model",
    "data": "Dataset",
    "performance": "Perf",
    "primary_metric": "Metric",
    "macs": "MACs",
    "params": "# Params",
}

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def _is_single_task_model(conf_model: DictConfig):
    conf_model_architecture_full = conf_model.architecture.full
    if conf_model_architecture_full is None:
        return False
    if conf_model_architecture_full.name is None:
        return False
    return True

@dataclass
class ExperimentSummary:
    id: str
    model: str
    task: str
    data: str
    data_format: str
    input_image_size: Tuple[int, int]
    checkpoint_path: str
    performance: float
    primary_metric: str
    macs: int
    params: int
    last_epoch: int
    best_epoch: int
    is_fx_retrain: bool


def _get_experiment_list(experiment_dir) -> List[ExperimentSummary]:

    experiment_dir = Path(experiment_dir)
    assert experiment_dir.exists(), f"Experiment root directory {str(experiment_dir)} does not exist!"

    experiment_dir_list_candidate = list(experiment_dir.glob("**/version_*"))
    experiment_dir_list = [run_dir for run_dir in experiment_dir_list_candidate
                           if (run_dir / "training_summary.json").exists()]

    experiment_list: List[ExperimentSummary] = []
    for run_dir in experiment_dir_list:
        summary = read_json(run_dir / "training_summary.json")
        
        if not summary['success']:
            continue
        
        hparam = OmegaConf.load(run_dir / "hparams.yaml")

        experiment_name = run_dir.parent.name
        best_epoch = summary['best_epoch']
        primary_metric = summary['primary_metric']
        model = str(hparam.model.name)

        experiment_list.append(
            ExperimentSummary(
                **
                {
                    "id": f"{experiment_name}/{run_dir.name}", "model": model, "task": hparam.model.task,
                    "data": hparam.data.name, "data_format": hparam.data.format,
                    "input_image_size": (hparam.logging.sample_input_size[0], hparam.logging.sample_input_size[1]),
                    "checkpoint_path":
                    [candidate for candidate in run_dir.glob("*.pt")
                     if candidate.suffix == '.pt' and 'best' in candidate.stem][0],
                    "performance": summary['valid_metrics'][str(best_epoch)][primary_metric],
                    "primary_metric": primary_metric,
                    "macs": summary['macs'],
                    "params": summary['params'],
                    "last_epoch": summary['last_epoch'],
                    "best_epoch": best_epoch,
                    "is_fx_retrain": (hparam.model.checkpoint.path.endswith('.pt') is not None)
                }
            )
        )

    return experiment_list


def _get_dataframe_from_experiment_list(experiment_list: List[ExperimentSummary]) -> pd.DataFrame:
    df = pd.DataFrame(experiment_list)
    return df


def get_dataframe(experiment_dir=OUTPUT_DIR) -> pd.DataFrame:
    experiment_list = _get_experiment_list(experiment_dir)
    df = _get_dataframe_from_experiment_list(experiment_list)
    return df


class ExperimentDataFrame:
    column_name_as = COLUMN_NAME_AS

    def __init__(self, headers, experiment_dir=OUTPUT_DIR) -> None:
        self.headers = headers

        self._raw_df: pd.DataFrame = get_dataframe(experiment_dir)
        self._df: pd.DataFrame = self._raw_df[self.headers]

    def _render(self, df: Optional[pd.DataFrame] = None):

        if df is None:
            df = self._df

        df = df.copy()
        df.is_fx_retrain.replace(to_replace={True: "⭕️", False: "-"}, inplace=True)
        df.rename(columns=self.column_name_as, inplace=True)
        return df

    @property
    def default(self):
        # To force copying df, you don't have to put `self._df` itself as target df
        return self._render()

    @property
    def default_no_render(self):
        return self._df

    def select_with_headers(self, headers: List[str]) -> pd.DataFrame:
        filtered_df = self._raw_df[headers]
        return self._render(filtered_df)

    @staticmethod
    def dropdown_selected(dropdown_input):
        return dropdown_input is not None and len(dropdown_input) > 0

    @staticmethod
    def slider_selected(slider_input):
        return slider_input is not None and slider_input > 0

    def filter_with(self, task, data, model, macs, params, ignore_compressed_model):
        queries = []

        if self.dropdown_selected(task):
            queries.append(f'`task` == "{task}"')

        if self.dropdown_selected(data):
            queries.append(f'`data` == "{data}"')

        if self.dropdown_selected(model):
            queries.append(f'`model` == "{model}"')

        if self.slider_selected(macs):
            queries.append(f'`macs` <= {macs * (10 ** 9)}')

        if self.slider_selected(params):
            queries.append(f'`params` <= {params * (10 ** 6)}')

        if ignore_compressed_model:
            queries.append('`is_fx_retrain` == False')

        if len(queries) == 0:
            return self.default

        query_str = " & ".join(queries)
        print(query_str)
        filtered_df = self._df.query(query_str)

        return self._render(filtered_df)

    def find_compression_info_with_id(self, id: str):
        filtered_df = self._raw_df.loc[self._raw_df['id'] == id]

        assert filtered_df.shape[0] == 1, f"No unique experiment found with the given id: {id}"

        model_name = filtered_df['id'].values[0]
        model_path = filtered_df['checkpoint_path'].values[0]

        input_image_size: Tuple[int, int] = filtered_df['input_image_size'].values[0]
        image_height, image_width = input_image_size
        compress_input_batch_size = 1
        compress_input_channels = 3
        compress_input_height = image_height
        compress_input_width = image_width

        return model_name, model_path, compress_input_batch_size, compress_input_channels, compress_input_height, compress_input_width
