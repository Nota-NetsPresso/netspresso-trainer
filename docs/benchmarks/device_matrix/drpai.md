# DRP-AI

The models in NetsPresso Trainer can be converted to DRP-AI format by NetsPresso's Launcher module. The converted DRP-AI model's performance can be measured on actual boards using NetsPresso's Benchmarker module. For more detailed information, please refer to [NetsPresso](https://netspresso.ai/).

Note that the latency value only measures the time for the model's computation and does not include the time on data preprocessing or postprocessing.

## Renesas RZ V2L

### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | ResNet18 | (224, 224) | 3 | 23.9425 | - | - | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 40.0319 | - | - | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 58.8135 | - | - | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 196.642 | - | - | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 412.707 | - | - | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 687.293 | - | - | onnx_opset=13 |
