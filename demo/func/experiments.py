from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
import pandas as pd
from omegaconf import OmegaConf, DictConfig

OUTPUT_DIR = Path("./outputs")

COLUMN_NAME_AS = {
    "is_fx_retrain": "🦴",
    "id": "Name",
    "model": "Model",
    "task": "Task",
    "data": "Dataset",
    "performance": "Perf",
    "primary_metric": "Metric",
    "macs": "MACs",
    "params": "# Params",
}


def _is_single_task_model(conf_model: DictConfig):
    conf_model_architecture_full = conf_model.architecture.full
    if conf_model_architecture_full is None:
        return False
    if conf_model_architecture_full.name is None:
        return False
    return True


def _get_model_name(conf_model: DictConfig):
    single_task_model = _is_single_task_model(conf_model)
    conf_model_sub = conf_model.architecture.full if single_task_model else conf_model.architecture.backbone
    model_name = conf_model_sub.name
    return model_name


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

    experiment_dir_list_candidate = [run_dir for run_dir in experiment_dir.glob("**/version_*")]
    experiment_dir_list = [run_dir for run_dir in experiment_dir_list_candidate
                           if (run_dir / "training_summary.ckpt").exists()]

    experiment_list: List[ExperimentSummary] = []
    for run_dir in experiment_dir_list:
        summary = torch.load(run_dir / "training_summary.ckpt")
        hparam = OmegaConf.load(run_dir / "hparams.yaml")

        experiment_name = run_dir.parent.name
        best_epoch = summary['best_epoch']
        primary_metric = summary['primary_metric']
        model = _get_model_name(hparam.model)

        experiment_list.append(
            ExperimentSummary(
                **
                {
                    "id": f"{experiment_name}/{run_dir.name}", "model": model, "task": hparam.model.task,
                    "data": hparam.data.name, "data_format": hparam.data.format,
                    "input_image_size": (hparam.augmentation.img_size, hparam.augmentation.img_size),
                    "checkpoint_path":
                    [candidate for candidate in run_dir.glob("*.pt")
                     if candidate.suffix == '.pt' and 'best' in candidate.stem][0],
                    "performance": summary['valid_metrics'][best_epoch][primary_metric],
                    "primary_metric": primary_metric,
                    "macs": summary['macs'],
                    "params": summary['params'],
                    "last_epoch": summary['last_epoch'],
                    "best_epoch": best_epoch,
                    "is_fx_retrain": (hparam.model.fx_model_checkpoint is not None)
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

    def __init__(self, experiment_dir=OUTPUT_DIR) -> None:
        self.experiment_dir = experiment_dir
        self._df = get_dataframe(self.experiment_dir)

    def _render(self, df: Optional[pd.DataFrame] = None):

        if df is None:
            df = self._df.copy()

        df.is_fx_retrain.replace(to_replace={True: "⭕️", False: "-"}, inplace=True)
        df.rename(columns=self.column_name_as, inplace=True)
        return df

    @property
    def dataframe(self):
        return self._df

    @property
    def dataframe_rendered(self):
        # To force copying df, you don't have to put `self._df` itself as target df
        return self._render()

    def filtered_with_headers(self, headers: List[str]) -> pd.DataFrame:
        filtered_df = self._df[headers].copy()
        return self._render(filtered_df)
