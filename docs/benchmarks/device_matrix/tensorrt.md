# TensorRT

The models in NetsPresso Trainer can be converted to TensorRT format by NetsPresso's Launcher module. During this process, you can specify the JetPack version. The converted TensorRT model's performance can be measured on actual boards using NetsPresso's Benchmarker module.

Using NetsPresso, you can utilize more various JetPack versions and devices than listed in this document. For more detailed information, please refer to [NetsPresso](https://netspresso.ai/).

Note that the latency value only measures the time for the model's computation and does not include the time on data preprocessing or postprocessing.

## JetPack 6.0

### Jetson Orin Nano

#### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 4.45334 | 23.0 | - | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 6.33559 | 12.0 | 1.0 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 9.07441 | 16.0 | 1.0 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 11.4235 | 23.0 | 1.0 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 1.56278 | 4.0 | - | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 2.51209 | 11.0 | - | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 7.13308 | 18.0 | - |  |
| Classification | ResNet18 | (224, 224) | 3 | 1.99792 | 23.0 | - | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 3.48368 | 42.0 | - | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 4.80201 | 48.0 | - | onnx_opset=13 |
| Classification | ViT-tiny | (224, 224) | 3 | 3.11549 | 11.0 | - | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 7.32771 | 25.0 | - | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 22.6924 | 69.0 | - | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 15.508 | 29.0 | - | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 32.276 | 25.0 | - | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 53.4317 | 69.0 | - | onnx_opset=13 |

## JetPack 5.0.1

### Jetson AGX Orin

#### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 1.35066 | 23.0 | 302.0 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 2.1774 | 964.0 | 920.0 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 2.83159 | 969.0 | 921.0 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 3.30286 | 977.0 | 920.0 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 0.721624 | 4.0 | 302.0 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 0.919789 | 10.0 | 302.0 | onnx_opset=13 |
| Classification | ResNet18 | (224, 224) | 3 | 0.520046 | 23.0 | 302.0 | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 0.87017 | 42.0 | 301.0 | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 1.1421 | 48.0 | 302.0 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 1.87514 | 855.0 | 836.0 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 3.18338 | 31.0 | 301.0 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 6.2141 | 68.0 | 302.0 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 9.27001 | 130.0 | 302.0 | onnx_opset=13 |

## JetPack 4.6

### Jetson Nano

#### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 20.3864 | 695.0 | 601.0 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 22.5372 | 692.0 | 603.0 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 32.5569 | 691.0 | 602.0 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 42.1082 | 691.0 | 602.0 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 5.0241 | 692.0 | 603.0 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 11.278 | 692.0 | 600.0 | onnx_opset=13 |
| Classification | ResNet18 | (224, 224) | 3 | 10.8578 | 693.0 | 601.0 | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 19.5193 | 691.0 | 602.0 | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 29.2816 | 690.0 | 600.0 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 36.4498 | 694.0 | 603.0 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 95.8883 | 693.0 | 600.0 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 224.517 | 692.0 | 602.0 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 415.363 | 691.0 | 602.0 | onnx_opset=13 |

### Jetson NX

#### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 3.76573 | 893.0 | 886.0 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 5.13436 | 894.0 | 888.0 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 6.67466 | 893.0 | 888.0 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 8.16476 | 891.0 | 886.0 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 1.24565 | 881.0 | 886.0 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 1.96175 | 893.0 | 887.0 | onnx_opset=13 |
| Classification | ResNet18 | (224, 224) | 3 | 1.78444 | 893.0 | 887.0 | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 3.28956 | 887.0 | 886.0 | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 4.16903 | 893.0 | 887.0 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 6.80566 | 895.0 | 886.0 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 13.7659 | 892.0 | 887.0 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 29.2506 | 892.0 | 887.0 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 49.0844 | 896.0 | 888.0 | onnx_opset=13 |

### Jetson TX2

#### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 8.02968 | 727.0 | 657.0 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 10.6572 | 720.0 | 657.0 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 13.8955 | 721.0 | 657.0 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 17.8804 | 721.0 | 657.0 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 3.64664 | 722.0 | 657.0 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 4.96121 | 721.0 | 656.0 | onnx_opset=13 |
| Classification | ResNet18 | (224, 224) | 3 | 4.18611 | 720.0 | 656.0 | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 7.52092 | 722.0 | 657.0 | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 11.0185 | 718.0 | 656.0 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 14.3191 | 723.0 | 657.0 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 35.8665 | 723.0 | 657.0 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 82.2585 | 719.0 | 655.0 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 153.366 | 719.0 | 657.0 | onnx_opset=13 |

### Jetson Xavier

#### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 2.6756 | 890.0 | 887.0 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 3.68271 | 891.0 | 887.0 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 4.87919 | 876.0 | 888.0 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 5.89479 | 895.0 | 886.0 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 1.00566 | 896.0 | 886.0 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 1.50451 | 886.0 | 887.0 | onnx_opset=13 |
| Classification | ResNet18 | (224, 224) | 3 | 1.10724 | 890.0 | 887.0 | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 1.99773 | 891.0 | 886.0 | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 2.83026 | 892.0 | 888.0 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 4.56521 | 894.0 | 886.0 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 8.83621 | 891.0 | 888.0 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 19.1866 | 892.0 | 887.0 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 30.6859 | 894.0 | 886.0 | onnx_opset=13 |
