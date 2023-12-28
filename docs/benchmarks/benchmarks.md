# Benchmarks

We are working on creating pretrained weights with NetsPresso Trainer and our own resources. We base training recipes on the official repositories or original papers as much as possible to replicate the performance of models.

For models that we have not yet trained with NetsPresso Trainer, we provide their pretrained weights from other awesome repositories. We have converted several models' weights into our own model architectures. We appreciate all the original authors and we also do our best to make other values.

Therefore, in the benchmark performance table of this section, a **Reproduced** status of True indicates performance obtained from our own training resources. In contrast, a False status means that the data is from original papers or repositories.

## Classification

[Download all weights (Google Drvie)](https://drive.google.com/drive/folders/15AoBl22hV8JStBu_CHD5WZd7dHBerKzf?usp=sharing)

| Dataset | Model | Weights | Resolution | Acc@1 | Acc@5 | Params | MACs | torch.fx | NetsPresso | Reproduced |
|---|---|---|---|---|---|---|---|---|---|---|
| ImageNet1K | [EfficientFormer-l1](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/model/efficientformer/efficientformer-l1-classification.yaml) | [download](https://drive.google.com/file/d/1I0SoTFs5AcI3mHpG_kDM2mW1PXDmG8X_/view?usp=drive_link) | 224x224 | 80.20 | - | 12.30M | 1.30G | Supported | Supported | False |
| ImageNet1K | [MixNet-s](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/model/mixnet/mixnet-s-classification.yaml) | - | 224x224 | 75.13 | - | - | - | Supported | Supported | False |
| ImageNet1K | [MobileNetV3-small](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/model/mobilenetv3/mobilenetv3-small-classification.yaml) | - | 224x224 | 67.67 | 87.40 | 2.50M | 0.03G | Supported | Supported | False |
| ImageNet1K | [MobileViT](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/model/mobilevit/mobilevit-s-classification.yaml) | [download](https://drive.google.com/file/d/1HF6iq1T0QSUqPViJobXx639xlBxkBHWd/view?usp=drive_link) | 224x224 | 78.40 | - | 5.60M | - | Supported | Supported | False |
| ImageNet1K | [ResNet50](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/model/resnet/resnet50-classification.yaml) | - | 224x224 | 79.61 | 94.67 | 25.56M | 2.62G | Supported | Supported | True |
| ImageNet1K | [ViT-tiny](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/model/vit/vit-classification.yaml) | [download](https://drive.google.com/file/d/1meGp4epqXcqplHnSkXHIVuvV2LYSaLFU/view?usp=drive_link) | 224x224 | 72.91 | - | 5.70M | - | Supported | Supported | False |

## Semantic segmentation

| Dataset | Model | Weights | Resolution | mIoU | Pixel acc | Params | MACs | torch.fx | NetsPresso | Reproduced |
|---|---|---|---|---|---|---|---|---|---|---|
| ADE20K | [SegFormer-b0](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/model/segformer/segformer-segmentation.yaml) | [download](https://drive.google.com/file/d/1QIvgBOwGKXfUS9ysDk3K9AkTAOaiyRXK/view?usp=drive_link) | 512x512 | 37.40 | - | 3.70M | 4.20G | Supported | Supported | False |
| Cityscapes | [PIDNet-s](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/model/pidnet/pidnet-s-segmentation.yaml) | [download](https://drive.google.com/file/d/16mGyzAJAgrefs7oXnxhGZaiG7T7Uriuf/view?usp=drive_link) | 2048x1024 | 78.80 | - | 7.60M | 23.80G | Supported | Supported | False |

## Object detection

| Dataset | Model | Weights | Resolution | mAP50 | mAP75 | mAP50:95 | Params | MACs | torch.fx | NetsPresso | Reproduced |
|---|---|---|---|---|---|---|---|---|---|---|---|
| COCO | [YOLOX-s](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/model/yolox/yolox-detection.yaml) | - | 640x640 | - | - | 40.50 | 9.00M | 13.40G | Supported | Supported | False |

## Acknowledgment

The original weight files which are not yet trained with NetsPresso Trainer are as follows.

- [EfficientFormer: apple/ml-cvnets](https://drive.google.com/file/d/11SbX-3cfqTOc247xKYubrAjBiUmr818y/view)
- [MobileViT: apple/ml-cvnets](https://apple.github.io/ml-cvnets/en/general/README-model-zoo.html#mobilevitv1-legacy)
- [ViT-tiny: apple/ml-cvnets](https://apple.github.io/ml-cvnets/en/general/README-model-zoo.html#classification-imagenet-1k)
- [SegFormer: (Hugging Face) nvidia](https://huggingface.co/nvidia/mit-b0) 
- [PIDNet: XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet#models)