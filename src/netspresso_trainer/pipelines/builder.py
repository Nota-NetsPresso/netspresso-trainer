from .registry import SUPPORTING_TASK_LIST, TASK_PIPELINE


def build_pipeline(conf, task, model_name, model, devices,
                   train_dataloader, eval_dataloader, class_map, logging_dir,
                   is_graphmodule_training, profile=False):
    assert task in SUPPORTING_TASK_LIST, f"No such task! (task: {task})"

    task_pipeline = TASK_PIPELINE[task]

    trainer = task_pipeline(conf, task, model_name, model, devices,
                            train_dataloader, eval_dataloader, class_map, logging_dir,
                            is_graphmodule_training=is_graphmodule_training, profile=profile)

    return trainer
