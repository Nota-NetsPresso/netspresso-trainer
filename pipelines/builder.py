from pipelines.registry import TASK_PIPELINE

def build_pipeline(args, task, model_name, model, devices, train_dataloader, eval_dataloader, class_map, profile, is_graphmodule_training):
    if task not in TASK_PIPELINE:
        raise AssertionError(f"No such task! (task: {task})")
    
    task_pipeline = TASK_PIPELINE[task]
    
    trainer = task_pipeline(args, task, model_name, model, devices,
                            train_dataloader, eval_dataloader, class_map,
                            profile=profile, is_graphmodule_training=is_graphmodule_training)

    return trainer