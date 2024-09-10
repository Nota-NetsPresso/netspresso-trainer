# OpenVINO

The models in NetsPresso Trainer can be converted to OpenVINO format by NetsPresso's Launcher module. The converted OpenVINO model's performance can be measured on actual boards using NetsPresso's Benchmarker module. For more detailed information, please refer to [NetsPresso](https://netspresso.ai/).

Note that the latency value only measures the time for the model's computation and does not include the time on data preprocessing or postprocessing.

## Intel Xeon W-2233

### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 5.65 | - | - | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 3.97 | - | - | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 6.17 | - | - | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 8.13 | - | - | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 2.36 | - | - | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 4.42 | - | - | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 11.69 | - | - | onnx_opset=13 |
| Classification | ResNet18 | (224, 224) | 3 | 5.94 | - | - | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 11.2 | - | - | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 13.34 | - | - | onnx_opset=13 |
| Classification | ViT-tiny | (224, 224) | 3 | 6.77 | - | - | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 17.95 | - | - | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 65.45 | - | - | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 44.81 | - | - | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 114.02 | - | - | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 227.48 | - | - | onnx_opset=13 |