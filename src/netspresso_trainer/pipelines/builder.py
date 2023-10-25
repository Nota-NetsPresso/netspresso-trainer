from .registry import TASK_PIPELINE, SUPPORTING_TASK_LIST


def build_pipeline(conf, task, model_name, model, devices, train_dataloader, eval_dataloader, class_map, is_graphmodule_training, profile=False):
    assert task in SUPPORTING_TASK_LIST, f"No such task! (task: {task})"

    task_ = task
    if task == 'detection':
        if conf.model.architecture.head.name in ['faster_rcnn']:
            task_ = 'detection-two-stage'
        else:
            task_ = 'detection-one-stage'

    task_pipeline = TASK_PIPELINE[task_]
    
    trainer = task_pipeline(conf, task, model_name, model, devices,
                            train_dataloader, eval_dataloader, class_map,
                            is_graphmodule_training=is_graphmodule_training, profile=profile)

    return trainer