import copy
import json
import os
from abc import ABC, abstractmethod
from ctypes import c_int
from dataclasses import asdict
from multiprocessing import Value
from pathlib import Path
from statistics import mean
from typing import Dict, List, Literal, Optional, final

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from ...loggers import START_EPOCH_ZERO_OR_ONE, build_logger
from ...losses import build_losses
from ...metrics import build_metrics
from ...optimizers import build_optimizer
from ...postprocessors import build_postprocessor
from ...schedulers import build_scheduler
from ...utils.checkpoint import load_checkpoint, save_checkpoint
from ...utils.fx import save_graphmodule
from ...utils.logger import yaml_for_logging
from ...utils.model_ema import build_ema
from ...utils.onnx import save_onnx
from ...utils.record import Timer, TrainingSummary
from ...utils.stats import get_params_and_macs

NUM_SAMPLES = 16


class BaseTaskProcessor(ABC):
    def __init__(self, devices):
        super(BaseTaskProcessor, self).__init__()
        self.devices = devices

    @abstractmethod
    def train_step(self, train_model, batch):
        raise NotImplementedError

    @abstractmethod
    def valid_step(self, eval_model, batch):
        raise NotImplementedError

    @abstractmethod
    def test_step(self, test_model, batch):
        raise NotImplementedError

    @abstractmethod
    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid']):
        raise NotImplementedError
