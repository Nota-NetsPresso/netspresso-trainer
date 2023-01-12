import os
from pathlib import Path
from collections import namedtuple
from typing import List
import time

from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .base import BasePipeline
from ..utils.search_api import ModelSearchServerHandler

from atomixnet.train_validate.train_validate import reduce_tensor
from atomixnet.train_validate.show_res import ShowResults
from atomixnet.others.common import get_device, AverageMeter
from atomixnet.train_validate.utils import *
from atomixnet.train_validate.cuda import Scaler
from atomixnet.dataset.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from service.csv_logger import CSVLogger
from service.image_logger import ImageLogger


class ClassificationPipeline(BasePipeline):
    def __init__(self, args, model, **kwargs):
        super(ClassificationPipeline, self).__init__(args, model, **kwargs)
        
    def train_one_epoch(self, one_epoch_result):
        for batch in self.dataloader:
            one_epoch_result.append(self.model(batch))
            
        return one_epoch_result
    

CKPT_DIR = "weights"
BEST_CKPT_NAME = "best.pt"
RESULT_CSV_PATH = "results.csv" # https://github.com/nota-github/trainer_cls_example/blob/netspresso/exp/results.csv
RESULT_IMAGE_DIR = "img" # https://github.com/nota-github/trainer_cls_example/tree/netspresso/exp/img

SAVE_EPOCH_INCREMENT = 1

class CustomTrainValidate():
    def __init__(self, epochs, optimizer, scheduler,
                 criterion, logger, output_dir,
                 rank, world_size, distributed, args,
                 preserved_idx=None,
                 idx_to_class=None,
                 amp_state_dict=None, ema_state_dict=None,
                 ):
        
        self.model_search_handler = ModelSearchServerHandler(args.train.project, token=args.train.token)
        
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
        self.amp_state_dict = amp_state_dict
        self.ema_state_dict = ema_state_dict

        self.args = args
        self.logger = logger
        
        self.output_dir = Path(output_dir)
        self.ckpt_dir = self.output_dir / CKPT_DIR
        self.ckpt_dir.mkdir(exist_ok=True)
        
        self.results = ShowResults(logger)
        
        csv_path = self.output_dir / RESULT_CSV_PATH
        self.csv_logger = CSVLogger(csv_path=csv_path)
        
        image_dir = self.output_dir / RESULT_IMAGE_DIR
        image_dir.mkdir(exist_ok=True)
        assert isinstance(idx_to_class, dict)
        self.image_logger = ImageLogger(image_dir=image_dir, idx_to_class=idx_to_class)

        self.rank = rank
        self.distributed = distributed
        self.world_size = world_size

        self.preserved_idx = preserved_idx

    # =============  training / finetuning =============

    def train(self, model, train_loader, val_loader, input = None, criterion=None, epochs=None, optimizer=None, scheduler=None, save_result=True, save_plot=False, start_epoch=0, use_model_ema=False, kd_wrapper=None, use_amp=False):
        """
            Function to call to train the model.
        """
        self.logger.info('-'*40)
        self.logger.info('==> Training...')
        
        criterion = self.criterion if criterion is None else criterion
        epochs = self.epochs if epochs is None else epochs
        optimizer = self.optimizer if optimizer is None else optimizer
        scheduler = self.scheduler if scheduler is None else scheduler

        acc = self._train(model, train_loader, val_loader, criterion, epochs, optimizer, scheduler, save_result, save_plot, start_epoch, use_model_ema, kd_wrapper, use_amp)
        if save_result:
            self.validate(model, val_loader, criterion, input, save_result=True, key='after_training', description='Performances trained model')
        if save_plot and self.rank==0:
            self.results.plot_acc(acc, self.output_dir, title="Acc vs epoch", filename=f"plot")
            
    def finetune(self, model, train_loader, val_loader, input = None, criterion=None, epochs=None, optimizer=None, scheduler=None, save_result=True, save_plot=False, start_epoch=0, use_model_ema=False, kd_wrapper=None, use_amp=False):
        """
            Function to call to finetune the model.
        """
        self.logger.info('-'*40)
        self.logger.info('==> Finetuning...')

        criterion = self.criterion if criterion is None else criterion
        epochs = self.epochs if epochs is None else epochs
        optimizer = self.optimizer if optimizer is None else optimizer
        scheduler = self.scheduler if scheduler is None else scheduler

        acc = self._train(model, train_loader, val_loader, criterion, epochs, optimizer, scheduler, save_result, save_plot, start_epoch, use_model_ema, kd_wrapper, use_amp)
        if save_result:
            self.validate(model, val_loader, criterion, input, save_result=True, key='after_finetune', description='Performances finetuned model')
        if save_plot and self.rank==0:
            self.results.plot_acc(acc, self.output_dir, title="Acc vs epoch", filename=f"plot_{self.args.arch}")

    def validate(self, model, val_loader, criterion, input=None, mute=True, save_result=True, key=None, description=None):
        self.logger.info('-'*40)
        self.logger.info('==> Validation...')
        top1 = self._validate(model, val_loader, criterion, mute)[0]
        if save_result and self.rank==0:
            assert key is not None, "key must be provided to save results"
            assert description is not None, "description must be provided to save results"
            assert input is not None, "input must be provided to save results"
            self.results.update_results(model, input, top1, key, description)

    # =============  Show results =============

    def show_results(self):
        self.results.show_results()

    # ========== Intern functions for training =========


    def _train(self, model, train_loader, val_loader, criterion, epochs, optimizer, scheduler, save_result, save_plot, start_epoch, use_model_ema=False, kd_wrapper=None, use_amp=False):
        epochs = self.epochs if epochs is None else epochs
        optimizer = self.optimizer if optimizer is None else optimizer
        scheduler = self.scheduler if scheduler is None else scheduler
        criterion = self.criterion if criterion is None else criterion
        return self._train_model(model, train_loader, val_loader, criterion, epochs, optimizer, scheduler, start_epoch, use_model_ema, kd_wrapper, use_amp)

    def _train_model(self, model, train_loader, val_loader, criterion, epochs, optimizer, scheduler, start_epoch, use_model_ema=False, kd_wrapper=None, use_amp=False):
        device = get_device(model)

        loss_scaler = None
        if use_amp:
            loss_scaler = Scaler()
            self.logger.info('Using AMP. Training in mixed precision.')
            if self.amp_state_dict is not None:
                loss_scaler.load_state_dict(self.amp_state_dict)
        else:
            self.logger.info('AMP not enabled. Training in float32.')

        # setup exponential moving average of model weights 
        model_ema = None
        if use_model_ema:
            decay = 0.9998
            self.logger.info(f'Using exp moving average of model weights, with decay {decay}')
            from atomixnet.train_validate.model_ema.model_ema import ModelEma
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(model, decay=decay)
            if self.ema_state_dict is not None:
                model_ema.load_state_dict(self.ema_state_dict)
            # if args.resume:
            #     load_checkpoint(model_ema.module, args.resume, use_ema=True)

        # setup distributed training
        if self.distributed:
            self.logger.info("Using DistributedDataParallel.")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(device)
            model = DDP(model, broadcast_buffers=True, device_ids=[self.rank])
            # NOTE: EMA model does not need to be wrapped by DDP

        # knowledge distillation
        if kd_wrapper is not None:
            self.logger.info(f'Using KD')

        best_top1_acc = 0
        best_top5_acc = 0
        epoch_best_top1_acc = 0
        # adjust the learning rate according to the checkpoint
        if start_epoch > 0:
            scheduler.step(epoch=start_epoch)
            
        # train the model
        # save model and reload before finetuning?
        epoch = start_epoch
        acc = []
        while epoch < epochs:
            
            elapsed_time = self.train_one_epoch(model, train_loader, criterion, epoch, optimizer, scheduler,
                                                loss_scaler=loss_scaler, device=device, kd_wrapper=kd_wrapper, model_ema=model_ema)

            if epoch == start_epoch:
                self.model_search_handler.report_elapsed_time_for_epoch(int(elapsed_time))

            valid_top1_acc, valid_top5_acc = self._validate(model, val_loader, criterion=criterion)
            acc.append(valid_top1_acc)
            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                epoch_best_top1_acc = epoch
                is_best = True

            if self.rank==0:
                save_checkpoint({
                    'epoch': epoch + SAVE_EPOCH_INCREMENT,
                    'state_dict': model.state_dict(),
                    'top1_acc': valid_top1_acc,
                    'optimizer': optimizer.state_dict(),
                    'model_ema': model_ema.state_dict() if model_ema is not None else None,
                    'preserved_idx': self.preserved_idx,
                    'amp': loss_scaler.state_dict() if use_amp else None
                }, is_best, self.ckpt_dir, filename_best=BEST_CKPT_NAME)

            epoch += 1
            self.logger.info("=> Best accuracy {:.3f} (at epoch {})".format(best_top1_acc, epoch_best_top1_acc)) 
            if epoch < epochs: self.logger.info("-"*10)

        # load the best epoch
        if self.distributed:
            torch.distributed.barrier() # make sure process 0 saved the model
        path_best = os.path.join(self.ckpt_dir, BEST_CKPT_NAME)
        checkpoint = torch.load(path_best, map_location=device if self.distributed else device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if use_amp:
            loss_scaler.load_state_dict(checkpoint['amp'])
            
        return acc
    
    def train_one_epoch(self, model, train_loader, criterion, epoch, opti, sched,
                        loss_scaler=None, device=None, kd_wrapper=None, model_ema=None):
        epoch_start_time = time.time()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')

        model.train()

        lrl = [param_group['lr'] for param_group in opti.param_groups]
        cur_lr = sum(lrl) / len(lrl)

        num_iter = len(train_loader)
        num_updates = epoch * num_iter

        if self.distributed:
            train_loader.sampler.set_epoch(epoch)   

        self.logger.info(f'Epoch[{epoch}]')
        self.logger.info(f'learning rate: {cur_lr:.7f}')
        for i, (images, target) in enumerate(tqdm(train_loader, leave=False)):

            if device != images.device:
                images = images.to(device)
                target = target.to(device)

            # compute gradient and do SGD step
            opti.zero_grad()

            # compute output
            with autocast():
                # If a Tensor from the autocast region is already casted, the cast is a no-op, and incurs no additional overhead.
                logits = model(images)
                loss = criterion(logits, target)
                
                # measure accuracy
                pred1, pred5 = accuracy(logits, target, topk=(1, 5))
                n = images.size(0)

                if self.distributed:
                    pred1 = reduce_tensor(pred1, self.world_size)
                    pred5 = reduce_tensor(pred5, self.world_size)

                top1.update(pred1.item(), n)
                top5.update(pred5.item(), n)

                # KD logic
                if kd_wrapper is not None:
                    # student probability calculation
                    prob_s = F.log_softmax(logits, dim=-1)

                    # teacher probability calculation
                    with torch.no_grad():
                        input_kd = kd_wrapper.normalize_input(images, model)
                        out_t = kd_wrapper.model(input_kd.detach())
                        prob_t = F.softmax(out_t, dim=-1)

                    # adding KL loss
                    loss += kd_wrapper.get_alpha() * F.kl_div(prob_s, prob_t, reduction='batchmean')

            # measure accuracy and record loss
            if self.distributed:
                reduced_loss = reduce_tensor(loss.data, self.world_size)
                losses.update(reduced_loss.item(), images.size(0)) 
            else:
                losses.update(loss.item(), images.size(0)) # accumulated loss

            if loss_scaler is not None:
                loss_scaler(loss, opti)
            else:
                loss.backward()
                opti.step()

            if model_ema is not None:
                model_ema.update(model)

            if self.distributed:
                torch.distributed.barrier()

            num_updates += 1
            sched.step_update(num_updates=num_updates)


        self.logger.info(f'training loss: {losses.avg:.7f}')
        
        self.csv_logger.update(train_accuracy=top1.avg)
        self.csv_logger.update(train_loss=losses.avg)
        self.csv_logger.update(epoch=epoch)
        sched.step(epoch=epoch+1)
        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        return elapsed_time


    # ================== validate ======================

    def _validate(self, model, val_loader, criterion, mute=False):
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')

        device = get_device(model)

        # switch to evaluation mode
        model.eval()
        with torch.no_grad():
            for _, (images, target) in enumerate(val_loader):
                if device != images.device:
                    images = images.to(device)
                    target = target.to(device)

                # compute output
                with autocast():
                    logits = model(images)
                
                loss = criterion(logits, target)
                # measure accuracy
                pred1, pred5 = accuracy(logits, target, topk=(1, 5))
                n = images.size(0)

                if self.distributed:
                    pred1 = reduce_tensor(pred1, self.world_size)
                    pred5 = reduce_tensor(pred5, self.world_size)

                top1.update(pred1.item(), n)
                top5.update(pred5.item(), n)

            if not mute:
                self.logger.info(f'validation acc: {top1.avg:.3f} [top1] | {top5.avg:.3f} [top5]')
                
            if self.distributed:
                reduced_loss = reduce_tensor(loss.data, self.world_size)
                losses.update(reduced_loss.item(), images.size(0)) 
            else:
                losses.update(loss.item(), images.size(0)) # accumulated loss

            self.csv_logger.update(valid_accuracy=top1.avg)
            self.csv_logger.update(valid_loss=losses.avg)

        return top1.avg, top5.avg
    
    def infer_with_result_image(self, model, val_loader):
        device = get_device(model)

        # switch to evaluation mode
        model.eval()
        with torch.no_grad():
            for _, (images, targets) in enumerate(val_loader):
                if device != images.device:
                    images = images.to(device)
                    targets = targets.to(device)

                # compute output
                with autocast():
                    logits = model(images)
                                
                _, preds = logits.topk(k=1, dim=1)
                preds = preds.t()
                
                self.image_logger.save_img(images, preds, targets,
                                           mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)