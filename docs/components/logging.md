# Logging

NetsPresso Trainer provides training results in a variety of multiple formats. As a following example, users can determine most of output formats through boolean flags, and can adjust the intervals of evaluations and checkpoint saves with a simple configuration.

```yaml
logging:
  project_id: ~
  output_dir: ./outputs
  tensorboard: true
  image: true
  stdout: true
  model_save_options:
    save_optimizer_state: true
    save_best_only: false
    save_criterion: loss # metric
    sample_input_size: [512, 512] # Used for flops and onnx export
    onnx_export_opset: 13 # Recommend in range [13, 17]
    validation_epoch: &validation_epoch 10
    save_checkpoint_epoch: *validation_epoch  # Multiplier of `validation_epoch`.
  metrics:
    metric_names: ~ # None for default settings
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
| `logging.image` | (bool) Whether to save the validation results. It is ignored if the task is `classification`. |
| `logging.stdout` | (bool) Whether to log the standard output. |
| `logging.model_save_options.save_optimizer_state` | (bool) Whether to save optimizer state with model checkpoint to resume training. |
| `logging.model_save_options.save_best_only` | (bool) Whether to only the best model. |
| `logging.model_save_options.save_criterion` | (str) Criterion to determine which checkpoint is considered the best. One of 'loss' or 'metric'. |
| `logging.model_save_options.sample_input_size` | (list[int]) The size of the sample input used for calculating FLOPs and exporting the model to ONNX format. |
| `logging.model_save_options.onnx_export_opset` | (int) The ONNX opset version to be used for model export |
| `logging.model_save_options.validation_epoch` | (int) Validation frequency in total training process. |
| `logging.model_save_options.save_checkpoint_epoch` | (int) Checkpoint saving frequency in total training process. |
| `logging.metrics.metric_names` | (list(str), optional) List of metric names to be logged. If not specified, default metrics for the task will be used. |