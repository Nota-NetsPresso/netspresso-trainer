# Overview

This section describes the architecture configuration design of models. NetsPresso Trainer prioritize model compression and device deployment, thus models are fulfills the following criteria:

- Compatible with torch.fx converting
- Can be compressed with pruning method provided in [NetsPresso](https://netspresso.ai)
- Efficient to be easily deployed at many edge devices.

To provide a wide range of models that meet these conditions in diverse forms, we define and use four fields for model definition: full, backbone, neck, and head. This approach allows users to utilize backbones, necks, and heads in desired configurations. For models that cannot be segmented into these three modules, we provide them in a full models.

```yaml
model:
  architecture:
    full: ~ # For full model which can't be separated to backbone, neck and head.
    backbone: ~ # Model backbone configuration.
    neck: ~ # Model neck configuration.
    head: ~ # Model head configuration.
```

## Pretrained weights

For now, we provide the pretrained weight from other awesome repositories. We have converted several models' weights into our own model architectures. In the near soon, we are planning to provide the pretrained weights directly trained from our resources.

We appreciate all the original authors and we also do our best to make other values.

[Download all weights (Google Drvie)](https://drive.google.com/drive/folders/15AoBl22hV8JStBu_CHD5WZd7dHBerKzf?usp=sharing)

| Family           | Model    | Link    | Origianl repository    |
| ------           | -----    | ----    | -------------------    |
| ResNet           | [`resnet50`](./config/model/resnet.yaml) | [Google Drive](https://drive.google.com/file/d/1xFfPcea8VyZ5KlegrIcSMUpRZ-FKOvKF/view?usp=drive_link) | [torchvision](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights) |
| ViT              | [`vit_tiny`](./config/model/vit.yaml) | [Google Drive](https://drive.google.com/file/d/1meGp4epqXcqplHnSkXHIVuvV2LYSaLFU/view?usp=drive_link) | [apple/ml-cvnets](https://apple.github.io/ml-cvnets/en/general/README-model-zoo.html#classification-imagenet-1k) |
| MobileViT        | [`mobilevit_s`](./config/model/mobilevit.yaml) | [Google Drive](https://drive.google.com/file/d/1HF6iq1T0QSUqPViJobXx639xlBxkBHWd/view?usp=drive_link) | [apple/ml-cvnets](https://apple.github.io/ml-cvnets/en/general/README-model-zoo.html#mobilevitv1-legacy) |
| SegFormer        | [`segformer`](./config/model/segformer.yaml) | [Google Drive](https://drive.google.com/file/d/1QIvgBOwGKXfUS9ysDk3K9AkTAOaiyRXK/view?usp=drive_link) | [(Hugging Face) nvidia](https://huggingface.co/nvidia/mit-b0) |
| EfficientForemer | [`efficientformer_l1_3000d`](./config/model/efficientformer.yaml) | [Google Drive](https://drive.google.com/file/d/1I0SoTFs5AcI3mHpG_kDM2mW1PXDmG8X_/view?usp=drive_link) | [snap-research/EfficientFormer](https://drive.google.com/file/d/11SbX-3cfqTOc247xKYubrAjBiUmr818y/view) |
| PIDNet           | [`pidnet_s`](./config/model/pidnet.yaml) | [Google Drive](https://drive.google.com/file/d/16mGyzAJAgrefs7oXnxhGZaiG7T7Uriuf/view?usp=drive_link) | [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet#models) |

## Field list

| Field <img width=200/> | Description |
|---|---|
| `full` | (dict) If the model does not distinctly separated to backbone, neck, and head, the model's details are defined under this field. If this field is not `None`, the `backbone`, `neck`, and `head` fields are ignored. |
| `backbone` | (dict) This field defines the model's backbone, applicable only when the `full` field is `None`. |
| `neck` | (dict) This field defines the model's neck, applicable only when the `full` field is `None`. This can be `None` anytime because the necessity of the neck module may vary depending on the task. |
| `head` | (dict) This field defines the model's head, applicable only when the `full` field is `None`. |
