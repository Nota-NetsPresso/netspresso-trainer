# Environment

The environment configuration contains values that determine the training environment, such as the number of workers for multi-processing and the GPU ids to be used. The following yaml is the environment configuration example.

```yaml
environment: 
  seed: 1
  batch_size: 8
  num_workers: 4 
  gpus: 0, 1, 2, 3
```

## Field list

| Field <img width=200/> | Description |
|---|---|
| `environment.seed` | (int) Random seed. |
| `environment.batch_size` | (int) The number of samples in single batch input. |
| `environment.num_workers` | (int) The number of multi-processing workers to be used by the data loader. |
| `environment.gpus` | (str) GPU ids to use, this should be separated by commas. |