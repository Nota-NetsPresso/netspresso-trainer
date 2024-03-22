from .registry import SUPPORTING_TASK_LIST, TASK_PROCESSOR, PIPELINES
from ..postprocessors import build_postprocessor


def build_pipeline(pipeline_type, conf, task, model_name, model, devices,
                   train_dataloader, eval_dataloader, class_map, logging_dir,
                   is_graphmodule_training, profile=False):
    assert task in SUPPORTING_TASK_LIST, f"No such task! (task: {task})"

    postprocessor = build_postprocessor(task, conf.model)
    task_processor = TASK_PROCESSOR[task](postprocessor, devices, conf.distributed)

    if pipeline_type == 'train':
        pipeline = PIPELINES[pipeline_type](conf, task, task_processor, model_name, model, devices,
                                            train_dataloader, eval_dataloader, class_map, logging_dir,
                                            is_graphmodule_training=is_graphmodule_training, profile=profile)

    return pipeline
