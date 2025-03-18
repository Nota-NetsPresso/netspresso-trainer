# Logging

NetsPresso Trainer provides training results in a variety of multiple formats. As a following example, users can determine most of output formats through boolean flags, and can adjust the intervals of evaluations and checkpoint saves with a simple configuration.

```yaml
logging:
  project_id: ~
  output_dir: ./outputs
  tensorboard: true
  mlflow: true
  image: true
  stdout: true
  model_save_options:
    save_optimizer_state: true
    save_best_only: false
    best_model_criterion: loss # metric
    sample_input_size: [512, 512] # Used for flops and onnx export
    onnx_export_opset: 13 # Recommend in range [13, 17]
    validation_epoch: &validation_epoch 10
    save_checkpoint_epoch: *validation_epoch  # Multiplier of `validation_epoch`.
  metrics:
    classwise_analysis: False
    metric_names: ~ # None for default settings
```

## Tensorboard

We provide basic tensorboard to track your training status. Run the tensorboard with the following command: 

```bash
tensorboard --logdir ./outputs --port 50001 --bind_all
```

Note that the default directory of saving result will be `./outputs` directory.
The port number `50001` is same with the port forwarded in example docker setup. You can change with any port number available in your environment.

## MLflow

### What is MLflow?
MLflow is an open-source platform designed to streamline and optimize the machine learning (ML) lifecycle. It helps ML practitioners and teams efficiently manage experiments, track model performance, and ensure reproducibility across different environments. Whether you're working locally or deploying models on cloud platforms like AWS, Azure, or Databricks, MLflow provides the necessary tools to simplify ML operations.
[🔗 Official MLflow GitHub Repository](https://github.com/mlflow/mlflow)

### How to Install MLflow

Getting started with MLflow is easy! Install it using pip:
```bash
pip install mlflow
```

### How to Run the MLflow Tracking Server Anywhere

MLflow is highly flexible and can be deployed in multiple environments, including:

- ✅ Local development
- ✅ Amazon SageMaker
- ✅ Amazon EC2
- ✅ Azure ML
- ✅ Databricks

For a step-by-step setup guide, visit the official MLflow Documentation.

💡 Why Use MLflow?

  - 📊 Experiment Tracking – Log, compare, and visualize ML runs effortlessly.
  - 🔄 Model Management – Register, version, and deploy models seamlessly.
  - 🔗 Platform Agnostic – Works with any ML framework (TensorFlow, PyTorch, Scikit-Learn, etc.).

### How to Log to the MLflow tracking server

To connect to your MLflow tracking server, set the required environment variables:
```bash
export MLFLOW_TRACKING_URI=<your/mlflow/tracking/server/uri>
export MLFLOW_EXPERIMENT_NAME=<your/experiment/name>  # default: `Default`
```
Then, enable MLflow logging by updating your logging.yaml file.

## Field list

| Field <img width=200/> | Description |
|---|---|
| `logging.project_id` | (str) Project name to save the experiment. If `None`, it is set as `{task}_{model}` (e.g. `segmentation_segformer`). |
| `logging.output_dir` | (str) Root directory for saving the experiment. Default location is `./outputs`. |
| `logging.tensorboard` | (bool) Whether to use the tensorboard. |
| `logging.mlflow` | (bool) Whether to use the mlflow. |
| `logging.image` | (bool) Whether to save the validation results. It is ignored if the task is `classification`. |
| `logging.stdout` | (bool) Whether to log the standard output. |
| `logging.model_save_options.save_optimizer_state` | (bool) Whether to save optimizer state with model checkpoint to resume training. |
| `logging.model_save_options.save_best_only` | (bool) Whether to only the best model. |
| `logging.model_save_options.best_model_criterion` | (str) Criterion to determine which checkpoint is considered the best. One of 'loss' or 'metric'. |
| `logging.model_save_options.sample_input_size` | (list[int]) The size of the sample input used for calculating FLOPs and exporting the model to ONNX format. |
| `logging.model_save_options.onnx_export_opset` | (int) The ONNX opset version to be used for model export |
| `logging.model_save_options.validation_epoch` | (int) Validation frequency in total training process. |
| `logging.model_save_options.save_checkpoint_epoch` | (int) Checkpoint saving frequency in total training process. |
| `logging.metrics.classwise_analysis` | (bool) Whether to perform class-wise analysis of metrics during validation. |
| `logging.metrics.metric_names` | (list(str), optional) List of metric names to be logged. If not specified, default metrics for the task will be used. |