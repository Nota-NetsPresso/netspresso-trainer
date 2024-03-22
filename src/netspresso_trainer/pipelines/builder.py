import torch.distributed as dist

from .registry import SUPPORTING_TASK_LIST, TASK_PROCESSOR, PIPELINES
from ..postprocessors import build_postprocessor
from ..loggers import build_logger
from .train import NUM_SAMPLES


def build_pipeline(pipeline_type, conf, task, model_name, model, devices,
                   train_dataloader, eval_dataloader, class_map, logging_dir,
                   is_graphmodule_training, profile=False):
    assert task in SUPPORTING_TASK_LIST, f"No such task! (task: {task})"

    postprocessor = build_postprocessor(task, conf.model)
    task_processor = TASK_PROCESSOR[task](postprocessor, devices, conf.distributed)

    if pipeline_type == 'train':
        single_gpu_or_rank_zero = (not conf.distributed) or (conf.distributed and dist.get_rank() == 0)
        train_step_per_epoch = len(train_dataloader)
        train_logger = None
        if single_gpu_or_rank_zero:
            train_logger = build_logger(conf, task, model_name,
                                        step_per_epoch=train_step_per_epoch,
                                        class_map=class_map,
                                        num_sample_images=NUM_SAMPLES,
                                        result_dir=logging_dir,)
        pipeline = PIPELINES[pipeline_type](conf, task, task_processor, model_name, model,
                                            train_dataloader, eval_dataloader, single_gpu_or_rank_zero,
                                            is_graphmodule_training=is_graphmodule_training, profile=profile, logger=train_logger)

    return pipeline
