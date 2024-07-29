# TensorRT

The models in NetsPresso Trainer can be converted to TensorRT format by NetsPresso's Launcher module. During this process, you can specify the JetPack version. The converted TensorRT model's performance can be measured on actual boards using NetsPresso's Benchmarker module.

Using NetsPresso, you can utilize more various JetPack versions and devices than listed in this document. For more detailed information, please refer to [NetsPresso](https://netspresso.ai/).

## JetPack 6.0

### Jetson Orin Nano

| Task | Model | Input shape | Classes | Compression ratio | Launcher | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 0.5 | ❌ |  |  |  | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 0.5 | ✅ | 6.6357 | 8.0 | 1.0 | onnx_opset=13 |
| Classification | MobileNetV3-s | (224, 224) | 3 | 0.5 | ❌ |  |  |  | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 0.5 | ❌ |  |  |  | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 0.5 | ❌ |  |  |  | onnx_opset=13 |
| Classification | ViT-tiny | (224, 224) | 3 | 0.5 | ❌ |  |  |  | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 0.5 | ❌ |  |  |  | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 0.5 | ❌ |  |  |  | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 0.5 | ❌ |  |  |  | onnx_opset=13 |

## JetPack 5.0.1

### Jetson AGX Orin

| Task | Model | Input shape | Classes | Compression ratio | Launcher | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 0.5 | ✅ | 1.3629 | 13.0 | 302.0 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 0.5 | ✅ | 2.2201 | 963.0 | 921.0 | onnx_opset=13 |
| Classification | MobileNetV3-s | (224, 224) | 3 | 0.5 | ✅ | 0.6396 | 2.0 | 301.0 | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 0.5 | ❌ | - | - | - | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 0.5 | ✅ | 0.9976 | 13.0 | 301.0 | onnx_opset=13 |
| Classification | ViT-tiny | (224, 224) | 3 | 0.5 | ❌ | - | - | - | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 0.5 | ✅ | 1.1214 | 6.0 | 302.0 | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 0.5 | ❌ | - | - | - | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 0.5 | ✅ | 1.1699 | 6.0 | 302.0 | onnx_opset=13 |
