# Logging

NetsPresso Trainer provides training results in a variety multiple formats. As a following example, users can decide most of output formats through boolean flags, and adjust the interval of evaluations and checkpoint save with simple configuration.

```yaml
logging:
  project_id: ~
  output_dir: ./outputs
  tensorboard: true
  csv: true
  image: true
  stdout: true
  save_optimizer_state: true
  validation_epoch: &validation_epoch 5
  save_checkpoint_epoch: *validation_epoch  # Multiplier of `validation_epoch`.
```

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
| `logging.project_id` | (str) Project name to save the experiment. If `None`, it is set as `{task}_{model}` (e.g. `segmentation_segformer`). |
| `logging.output_dir` | (str) Root directory for saving the experiment. Default location is `./outputs`. |
| `logging.tensorboard` | (bool) Whether to use the tensorboard. |
| `logging.csv` | (bool) Whether to save the result as csv format. |
| `logging.image` | (bool) Whether to save the validation results. It is ignored if the task is `classification`. |
| `logging.stdout` | (bool) Whether to log the standard output. |
| `logging.save_optimizer_state` | (bool) Whether to save optimizer state with model checkpoint to resume training. |
| `logging.validation_epoch` | (int) Validation frequency in total training process. |
| `logging.save_checkpoint_epoch` | (int) Checkpoint saving frequency in total training process. |