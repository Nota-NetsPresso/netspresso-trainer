## Tensorboard

We provide basic tensorboard to track your training status. Run the tensorboard with the following command: 

```bash
tensorboard --logdir ./outputs --port 50001 --bind_all
```

Note that the default directory of saving result will be `./outputs` directory.
The port number `50001` is same with the port forwarded in example docker setup. You can change with any port number available in your environment.

## Field list

| Field <img width=200/> | Description |
|---|---|
| `logging.project_id` | (str) project name to save the experiment. If None, it is set as `{task}_{model}` (e.g. `segmentation_segformer`)|
| `logging.output_dir` | (str) root directory for saving the experiment. Default location is `./outputs`|
| `logging.tensorboard` | (bool) whether to use the tensorboard or not |
| `logging.csv` | (bool) whether to save the result as csv format or not |
| `logging.image` | (bool) whether to save the validation results or not. It is ignored if the task is `classification`. |
| `logging.stdout` | (bool) whether to log the result with standard output or not. |
| `logging.save_optimizer_state` | (bool) switch for saving optimizer state with model checkpoint to resume training |
| `logging.validation_epoch` | (int) validation frequency in total training process |
| `logging.save_checkpoint_epoch` | (int) checkpoint saving frequency in total training process |