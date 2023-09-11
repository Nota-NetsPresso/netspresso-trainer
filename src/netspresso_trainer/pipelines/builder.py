from .registry import TASK_PIPELINE


def build_pipeline(conf, task, model_name, model, devices, train_dataloader, eval_dataloader, class_map, is_graphmodule_training, profile=False):
    assert task in TASK_PIPELINE, f"No such task! (task: {task})"
    
    task_pipeline = TASK_PIPELINE[task]
    
    trainer = task_pipeline(conf, task, model_name, model, devices,
                            train_dataloader, eval_dataloader, class_map,
                            is_graphmodule_training=is_graphmodule_training, profile=profile)

    return trainer