# Overview

This section describes the **architecture configuration design** of models. For details on NetsPresso Trainer's model configuration excluding the architecture, please refer to the [model components page](../components/model/overview.md).

NetsPresso Trainer prioritize model compression and device deployment, thus models fulfill the following criteria:

- Compatible with torch.fx converting.
- Can be compressed by pruning method provided in [NetsPresso](https://netspresso.ai).
- Can be easily deployed at many edge devices.

To provide a wide range of models that meet these conditions in diverse forms, we define and use four fields for model definition: full, backbone, neck, and head. This approach allows users to utilize backbones, necks, and heads in desired configurations. For models that cannot be segmented into these three modules, we provide them in a full models.

```yaml
model:
  architecture:
    full: ~ # For full model which can't be separated to backbone, neck and head.
    backbone: ~ # Model backbone configuration.
    neck: ~ # Model neck configuration.
    head: ~ # Model head configuration.
```

## Field list

| Field <img width=200/> | Description |
|---|---|
| `full` | (dict) If the model does not distinctly separated to backbone, neck, and head, the model's details are defined under this field. If this field is not `None`, the `backbone`, `neck`, and `head` fields are ignored. |
| `backbone` | (dict) This field defines the model's backbone, applicable only when the `full` field is `None`. |
| `neck` | (dict) This field defines the model's neck, applicable only when the `full` field is `None`. This can be `None` anytime because the necessity of the neck module may vary depending on the task. |
| `head` | (dict) This field defines the model's head, applicable only when the `full` field is `None`. |
