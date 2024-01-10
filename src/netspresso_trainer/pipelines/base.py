import copy
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, Literal, final

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from ..loggers import START_EPOCH_ZERO_OR_ONE, build_logger
from ..losses import build_losses
from ..metrics import build_metrics
from ..optimizers import build_optimizer
from ..postprocessors import build_postprocessor
from ..schedulers import build_scheduler
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.fx import save_graphmodule
from ..utils.logger import yaml_for_logging
from ..utils.onnx import save_onnx
from ..utils.record import Timer, TrainingSummary
from ..utils.stats import get_params_and_macs

NUM_SAMPLES = 16


class BasePipeline(ABC):
    def __init__(self, conf, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map, logging_dir,
                 is_graphmodule_training=False, profile=False):
        super(BasePipeline, self).__init__()
        self.conf = conf
        self.task = task
        self.model_name = model_name
        self.save_dtype = next(model.parameters()).dtype
        self.model = model.float()
        self.devices = devices
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_step_per_epoch = len(train_dataloader)
        self.training_history: Dict[int, Dict[
            Literal['train_losses', 'valid_losses', 'train_metrics', 'valid_metrics'], Dict[str, float]
        ]] = {}

        self.timer = Timer()

        self.loss_factory = None
        self.metric_factory = None
        self.optimizer = None
        self.start_epoch_at_one = bool(START_EPOCH_ZERO_OR_ONE)
        self.start_epoch = int(self.start_epoch_at_one)

        self.ignore_index = None
        self.num_classes = None

        self.profile = profile  # TODO: provide torch_tb_profiler for training
        self.is_graphmodule_training = is_graphmodule_training
        self.save_optimizer_state = self.conf.logging.save_optimizer_state

        self.single_gpu_or_rank_zero = (not self.conf.distributed) or (self.conf.distributed and dist.get_rank() == 0)

        if self.single_gpu_or_rank_zero:
            self.train_logger = build_logger(
                self.conf, self.task, self.model_name,
                step_per_epoch=self.train_step_per_epoch,
                class_map=class_map,
                num_sample_images=NUM_SAMPLES,
                result_dir=logging_dir,
            )

    @final
    def _is_ready(self):
        assert self.model is not None, "`self.model` is not defined!"
        assert self.optimizer is not None, "`self.optimizer` is not defined!"
        """Append here if you need more assertion checks!"""
        assert self.conf.logging.save_checkpoint_epoch % self.conf.logging.validation_epoch == 0, \
            "`save_checkpoint_epoch` should be the multiplier of `validation_epoch`."
        return True

    def set_train(self):

        assert self.model is not None
        self.optimizer = build_optimizer(self.model,
                                         optimizer_conf=self.conf.training.optimizer)
        self.scheduler, _ = build_scheduler(self.optimizer, self.conf.training)
        self.loss_factory = build_losses(self.conf.model, ignore_index=self.ignore_index)
        self.metric_factory = build_metrics(self.task, self.conf.model, ignore_index=self.ignore_index, num_classes=self.num_classes)
        self.postprocessor = build_postprocessor(self.task, self.conf.model)
        resume_optimizer_checkpoint = self.conf.model.resume_optimizer_checkpoint
        if resume_optimizer_checkpoint is not None:
            resume_optimizer_checkpoint = Path(resume_optimizer_checkpoint)
            if not resume_optimizer_checkpoint.exists():
                logger.warning(f"Traning summary checkpoint path {str(resume_optimizer_checkpoint)} is not found!"
                               f"Skip loading the previous history and trainer will be started from the beginning")
                return

            optimizer_dict = torch.load(resume_optimizer_checkpoint, map_location='cpu')
            optimizer_state_dict = optimizer_dict['optimizer']
            start_epoch = optimizer_dict['last_epoch'] + 1  # Start from the next to the end of last training
            start_epoch_at_one = optimizer_dict['start_epoch_at_one']

            self.optimizer.load_state_dict(optimizer_state_dict)
            self.scheduler.step(epoch=start_epoch)

            self.start_epoch_at_one = start_epoch_at_one
            self.start_epoch = start_epoch
            logger.info(f"Resume training from {str(resume_optimizer_checkpoint)}. Start training at epoch: {self.start_epoch}")

    def epoch_with_valid_logging(self, epoch: int):
        validation_freq = self.conf.logging.validation_epoch
        last_epoch = epoch == (self.conf.training.epochs + self.start_epoch_at_one - 1)
        return (epoch % validation_freq == self.start_epoch_at_one % validation_freq) or last_epoch

    def epoch_with_checkpoint_saving(self, epoch: int):
        checkpoint_freq = self.conf.logging.save_checkpoint_epoch
        last_epoch = epoch == (self.conf.training.epochs + self.start_epoch_at_one - 1)
        return (epoch % checkpoint_freq == self.start_epoch_at_one % checkpoint_freq) or last_epoch

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def valid_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid']):
        raise NotImplementedError

    @property
    def learning_rate(self):
        return mean([param_group['lr'] for param_group in self.optimizer.param_groups])

    @property
    def train_loss(self):
        return self.loss_factory.result('train').get('total').avg

    @property
    def valid_loss(self):
        return self.loss_factory.result('valid').get('total').avg

    @property
    def sample_input(self):
        return torch.randn((1, 3, self.conf.augmentation.img_size, self.conf.augmentation.img_size))

    def train(self):
        if self.single_gpu_or_rank_zero:
            logger.debug(f"Training configuration:\n{yaml_for_logging(self.conf)}")
            logger.info("-" * 40)

        self.timer.start_record(name='train_all')
        self._is_ready()

        num_epoch = -1
        try:
            for num_epoch in range(self.start_epoch, self.conf.training.epochs + self.start_epoch_at_one):
                self.timer.start_record(name=f'train_epoch_{num_epoch}')
                self.loss_factory.reset_values()
                self.metric_factory.reset_values()

                self.train_one_epoch()

                with_valid_logging = self.epoch_with_valid_logging(num_epoch)
                with_checkpoint_saving = self.epoch_with_checkpoint_saving(num_epoch)
                # FIXME: multi-gpu sample counting & validation
                valid_samples = self.validate() if with_valid_logging else None

                self.timer.end_record(name=f'train_epoch_{num_epoch}')
                time_for_epoch = self.timer.get(name=f'train_epoch_{num_epoch}', as_pop=False)

                if self.single_gpu_or_rank_zero:
                    self.log_end_epoch(epoch=num_epoch,
                                       time_for_epoch=time_for_epoch,
                                       valid_samples=valid_samples,
                                       valid_logging=with_valid_logging)
                    if with_checkpoint_saving:
                        assert with_valid_logging
                        self.save_checkpoint(epoch=num_epoch)
                        self.save_summary()
                    logger.info("-" * 40)

                self.scheduler.step()  # call after reporting the current `learning_rate`

            self.timer.end_record(name='train_all')
            total_train_time = self.timer.get(name='train_all', as_pop=False)
            logger.info(f"Total time: {total_train_time:.2f} s")

            if self.single_gpu_or_rank_zero:
                self.train_logger.log_end_of_traning(final_metrics={'time_for_last_epoch': time_for_epoch})
                self.save_summary(end_training=True)
        except KeyboardInterrupt as e:
            # TODO: add independent procedure for KeyboardInterupt
            logger.error("Keyboard interrupt detected! Try saving the current checkpoint...")
            if self.single_gpu_or_rank_zero:
                self.save_checkpoint(epoch=num_epoch)
                self.save_summary()
            raise e
        except Exception as e:
            logger.error(str(e))
            raise e

    def train_one_epoch(self):
        outputs = []
        for _idx, batch in enumerate(tqdm(self.train_dataloader, leave=False)):
            out = self.train_step(batch)
            outputs.append(out)
        self.get_metric_with_all_outputs(outputs, phase='train')

    @torch.no_grad()
    def validate(self, num_samples=NUM_SAMPLES):
        num_returning_samples = 0
        returning_samples = []
        outputs = []
        for _idx, batch in enumerate(tqdm(self.eval_dataloader, leave=False)):
            out = self.valid_step(batch)
            if out is not None:
                outputs.append(out)
                if num_returning_samples < num_samples:
                    returning_samples.append(out)
                    num_returning_samples += len(out['pred'])
        self.get_metric_with_all_outputs(outputs, phase='valid')
        return returning_samples

    @torch.no_grad()
    def inference(self, test_dataset):
        returning_samples = []
        for _idx, batch in enumerate(tqdm(test_dataset, leave=False)):
            out = self.test_step(batch)
            returning_samples.append(out)
        return returning_samples

    def log_end_epoch(self, epoch, time_for_epoch, valid_samples=None, valid_logging=False):
        train_losses = self.loss_factory.result('train')
        train_metrics = self.metric_factory.result('train')

        valid_losses = self.loss_factory.result('valid') if valid_logging else None
        valid_metrics = self.metric_factory.result('valid') if valid_logging else None

        self.train_logger.update_epoch(epoch)
        self.train_logger.log(
            train_losses=train_losses,
            train_metrics=train_metrics,
            valid_losses=valid_losses,
            valid_metrics=valid_metrics,
            train_images=None,
            valid_images=valid_samples,
            learning_rate=self.learning_rate,
            elapsed_time=time_for_epoch
        )

        summary_record = {'train_losses': train_losses, 'train_metrics': train_metrics}
        if valid_logging:
            summary_record.update({'valid_losses': valid_losses, 'valid_metrics': valid_metrics})
        self.training_history.update({epoch: summary_record})

    def save_checkpoint(self, epoch: int):

        # Check whether the valid loss is minimum at this epoch
        valid_losses = {epoch: record['valid_losses'].get('total') for epoch, record in self.training_history.items()
                        if 'valid_losses' in record}
        best_epoch = min(valid_losses, key=valid_losses.get)
        save_best_model = best_epoch == epoch

        model = self.model.module if hasattr(self.model, 'module') else self.model
        if self.save_dtype == torch.float16:
            model = copy.deepcopy(model).type(self.save_dtype)
        logging_dir = self.train_logger.result_dir
        model_path = Path(logging_dir) / f"{self.task}_{self.model_name}_epoch_{epoch}.ext"
        best_model_path = Path(logging_dir) / f"{self.task}_{self.model_name}_best.ext"
        optimizer_path = Path(logging_dir) / f"{self.task}_{self.model_name}_epoch_{epoch}_optimzer.pth"

        if self.save_optimizer_state:
            optimizer = self.optimizer.module if hasattr(self.optimizer, 'module') else self.optimizer
            save_dict = {'optimizer': optimizer.state_dict(), 'start_epoch_at_one': self.start_epoch_at_one, 'last_epoch': epoch}
            torch.save(save_dict, optimizer_path)
            logger.debug(f"Optimizer state saved at {str(optimizer_path)}")

        if self.is_graphmodule_training:
            # Just save graphmodule checkpoint
            torch.save(model, model_path.with_suffix(".pt"))
            logger.debug(f"PyTorch FX model saved at {str(model_path.with_suffix('.pt'))}")
            if save_best_model:
                save_onnx(model, best_model_path.with_suffix(".onnx"), sample_input=self.sample_input.type(self.save_dtype))
                logger.info(f"ONNX model converting and saved at {str(best_model_path.with_suffix('.onnx'))}")
                torch.save(model, best_model_path.with_suffix(".pt"))
                logger.info(f"Best model saved at {str(best_model_path.with_suffix('.pt'))}")
            return
        pytorch_model_state_dict_path = model_path.with_suffix(".safetensors")
        save_checkpoint(model.state_dict(), pytorch_model_state_dict_path)
        logger.debug(f"PyTorch model saved at {str(pytorch_model_state_dict_path)}")
        if save_best_model:
            pytorch_best_model_state_dict_path = best_model_path.with_suffix(".safetensors")
            save_checkpoint(model.state_dict(), pytorch_best_model_state_dict_path)
            logger.info(f"Best model saved at {str(pytorch_best_model_state_dict_path)}")

            try:
                save_onnx(model, best_model_path.with_suffix(".onnx"), sample_input=self.sample_input.type(self.save_dtype))
                logger.info(f"ONNX model converting and saved at {str(best_model_path.with_suffix('.onnx'))}")

                save_graphmodule(model, (model_path.parent / f"{best_model_path.stem}_fx").with_suffix(".pt"))
                logger.info(f"PyTorch FX model tracing and saved at {str(best_model_path.with_suffix('.pt'))}")
            except Exception as e:
                logger.error(e)
                pass

    def save_summary(self, end_training=False):
        training_summary = TrainingSummary(
            total_epoch=self.conf.training.epochs,
            start_epoch_at_one=self.start_epoch_at_one,
            train_losses={epoch: record['train_losses'].get('total') for epoch, record in self.training_history.items()},
            valid_losses={epoch: record['valid_losses'].get('total') for epoch, record in self.training_history.items()
                          if 'valid_losses' in record},
            train_metrics={epoch: record['train_metrics'] for epoch, record in self.training_history.items()},
            valid_metrics={epoch: record['valid_metrics'] for epoch, record in self.training_history.items()
                           if 'valid_metrics' in record},
            metrics_list=self.metric_factory.metric_names,
            primary_metric=self.metric_factory.primary_metric,
        )
        if end_training:
            total_train_time = self.timer.get(name='train_all', as_pop=True)
            macs, params = get_params_and_macs(self.model, self.sample_input.float())
            logger.info(f"[Model stats] Params: {(params/1e6):.2f}M | MACs: {(macs/1e9):.2f}G")
            training_summary.total_train_time = total_train_time
            training_summary.macs = macs
            training_summary.params = params
            training_summary.success = True

        logging_dir = self.train_logger.result_dir
        summary_path = Path(logging_dir) / "training_summary.json"

        with open(summary_path, 'w') as f:
            json.dump(asdict(training_summary), f, indent=4)
        logger.info(f"Model training summary saved at {str(summary_path)}")

    def profile_one_epoch(self):
        PROFILE_WAIT = 1
        PROFILE_WARMUP = 1
        PROFILE_ACTIVE = 10
        PROFILE_REPEAT = 1
        _ = torch.ones(1).to(self.devices)
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=PROFILE_WAIT,
                                             warmup=PROFILE_WARMUP,
                                             active=PROFILE_ACTIVE,
                                             repeat=PROFILE_REPEAT),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True
        ) as prof:
            for idx, batch in enumerate(self.train_dataloader):
                if idx >= (PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE) * PROFILE_REPEAT:
                    break
                self.train_step(batch)
                prof.step()
