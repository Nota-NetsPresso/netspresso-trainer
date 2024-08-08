# TFLite

The models in NetsPresso Trainer can be converted to TFLite format by NetsPresso's Launcher module. The converted TFLite model's performance can be measured on actual boards using NetsPresso's Benchmarker module.

Using NetsPresso, you can utilize more various devices than listed in this document. For more detailed information, please refer to [NetsPresso](https://netspresso.ai/).

## RaspBerry PI 4B (FP16)

| Task | Model | Input shape | Classes | Compression ratio | Launcher | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 0.5 | ✅ | 273.961 | - | 59.9883 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 0.5 | ✅ | 80.0776 | - | 22.043 | onnx_opset=13 |
| Classification | MobileNetV3-s | (224, 224) | 3 | 0.5 | ✅ | 8.0722 | - | 5.25 | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 0.5 | ❌ |  |  |  | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 0.5 | ✅ | 187.1840 | - | 87.95 | onnx_opset=13 |
| Classification | ViT-tiny | (224, 224) | 3 | 0.5 | ✅ | 197.286 | - | 39.1172 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 0.5 | ✅ | 306.75 | - | 69.52 | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 0.5 | ✅ | 1645.6 | - | 277.234 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 0.5 | ✅ | 660.287 | - | 101.457 | onnx_opset=13 |

## 