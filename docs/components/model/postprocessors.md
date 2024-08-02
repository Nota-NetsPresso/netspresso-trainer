# Postprocessors

The postprocessor module is an essential component, designed to handle the output from deep learning models and apply necessary transformations to produce meaningful results. This module is particularly crucial for tasks such as object detection, where raw model outputs need to be processed into interpretable bounding boxes, confidence scores.

We currently provide the postprocessor in a model-wise rigid format. This will be improved in the future to allow for more flexible usage.

## Supporting postprocessors

The current postprocessor is automatically determined based on the task name and model name. Users can utilize it by filling in the necessary hyperparameters under the params field of the postprocessor.

### Classification

For classification, we don't any postprocessor settings yet.

```yaml
postprocessor: ~
```

### Segmentation

For segmentation, we don't any postprocessor settings yet.

```yaml
postprocessor: ~
```

### Detection

#### YOLOX

YOLOX performs box decoding and NMS (Non-Maximum Suppression) on its output. The necessary hyperparameters for these processes are set as follows:

```yaml
postprocessor: 
  params: 
    # postprocessor - decode
    score_thresh: 0.01
    # postprocessor - nms
    nms_thresh: 0.65
    class_agnostic: False
```
