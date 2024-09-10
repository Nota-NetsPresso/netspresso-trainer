# TFLite

The models in NetsPresso Trainer can be converted to TFLite format by NetsPresso's Launcher module. The converted TFLite model's performance can be measured on actual boards using NetsPresso's Benchmarker module. For more detailed information, please refer to [NetsPresso](https://netspresso.ai/).

Note that the latency value only measures the time for the model's computation and does not include the time on data preprocessing or postprocessing.

## Alif Ensemble E7 DevKit Gen 2

- Cortex-M55 + Ethos-U55
- Ethosu delegates

### INT8

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | MixNet-s | (224, 224) | 3 | 46.0708 | - | - | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 13.0257 | - | - | onnx_opset=13 |

## NXP iMX93

- Cortex-A55 + Ethos-U65
- Ethosu delegates

### INT8

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 21.6117 | - |  | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 20.1424 | - |   | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 27.7012 | - |  | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 37.3885 | - |  | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 3.21802 | - |  | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 7.05413 | - |  | onnx_opset=13 |
| Classification | MovileViT-s | - | - | - | - | - |  |
| Classification | ResNet18 | (224, 224) | 3 | 22.8633 | - |  | onnx_opset=13 |
| Classification | ResNet34 | - | - | - | - | - |  |
| Classification | ResNet50 | (224, 224) | 3 | 38.5456 | - |  | onnx_opset=13 |
| Classification | ViT-tiny | (224, 224) | 3 | 412.846 | - |  | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 87.6305 | - |  | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 2274.03 | - |  | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 133.775 | - |  | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 279.712 | - |  | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 503.778 | - |  | onnx_opset=13 |

## Raspberry Pi 5

### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 164.574 | - | 115.844 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 49.5337 | - | 38.4688 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 78.0415 | - | 64.1562 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 117.185 | - | 96.0156 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 4.077 | - | 20.6875 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 15.2487 | - | 55.2188 | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 228.259 | - | 115.234 | onnx_opset=13 |
| Classification | ResNet18 | (224, 224) | 3 | 55.2718 | - | 120.078 | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 98.0536 | - | 217.828 | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 130.835 | - | 278.656 | onnx_opset=13 |
| Classification | ViT-tiny | (224, 224) | 3 | 305.55 | - | 56.0 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 133.98 | - | 119.578 | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 1199.34 | - | 354.812 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 418.725 | - | 169.594 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 1176.87 | - | 357.891 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 2355.4 | - | 666.844 | onnx_opset=13 |

### INT8

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 30.6753 | - | 21.2969 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 80.8769 | - | 20.4375 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 82.4275 | - | 32.3906 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 73.1796 | - | 43.875 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 6.88714 | - | 3.53125 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 78.3648 | - | 11.5156 | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 224.878 | - | 30.3281 | onnx_opset=13 |
| Classification | ResNet18 | - | - | - | - | - |  |
| Classification | ResNet34 | - | - | - | - | - |  |
| Classification | ResNet50 | - | - | - | - | - |  |
| Classification | ViT-tiny | (224, 224) | 3 | 49.8123 | - | 12.0469 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 39.093 | - | 36.1719 | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 649.016 | - | 273.594 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 100.01 | - | 42.25 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 217.008 | - | 87.5 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 446.359 | - | 153.625 | onnx_opset=13 |

## Raspberry Pi 4B 

### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 439.871 | - | 119.438 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 108.501 | - | 40.3906 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 148.488 | - | 65.3008 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 231.303 | - | 98.332 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 14.1937 | - | 21.5352 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 43.9362 | - | 55.5117 | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 417.108 | - | 117.504 | onnx_opset=13 |
| Classification | ResNet18 | (224, 224) | 3 | 254.588 | - | 121.074 | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 486.278 | - | 220.734 | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 556.2 | - | 281.504 | onnx_opset=13 |
| Classification | ViT-tiny | (224, 224) | 3 | 286.019 | - | 57.5938 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 605.416 | - | 122.723 | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 2294.5 | - | 357.348 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 1488.71 | - | 171.43 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 4542.29 | - | 360.41 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 10087.7 | - | 669.797 | onnx_opset=13 |

### INT8

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 80.4513 | - | 23.1289 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 119.517 | - | 21.418 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 211.811 | - | 34.3984 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 276.174 | - | 45.75 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 18.4982 | - | 4.19531 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 57.0669 | - | 12.1953 | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 287.328 | - | 32.2891 | onnx_opset=13 |
| Classification | ResNet18 | - | - | - | - | - |  |
| Classification | ResNet34 | - | - | - | - | - |  |
| Classification | ResNet50 | - | - | - | - | - |  |
| Classification | ViT-tiny | (224, 224) | 3 | 197.057 | - | 12.3945 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 227.03 | - | 38.4648 | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 1393.21 | - | 275.422 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 533.657 | - | 46.2852 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 1468.42 | - | 87.9766 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 3133.25 | - | 154.941 | onnx_opset=13 |

## Raspberry Pi 3B PLUS

### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 1414.44 | - | 119.656 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 227.58 | - | 40.5938 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 348.734 | - | 65.5273 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 564.76 | - | 98.2148 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 28.9851 | - | 21.3633 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 114.964 | - | 55.293 | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 906.407 | - | 117.707 | onnx_opset=13 |
| Classification | ResNet18 | (224, 224) | 3 | 473.501 | - | 121.031 | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 864.672 | - | 220.539 | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 1091.6 | - | 281.406 | onnx_opset=13 |
| Classification | ViT-tiny | (224, 224) | 3 | 841.007 | - | 57.5156 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 1139.91 | - | 122.859 | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 5178.05 | - | 350.0 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 2881.02 | - | 171.043 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 7420.01 | - | 360.473 | onnx_opset=13 |

### INT8

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 156.684 | - | 23.0898 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 125.0 | - | 14.2891 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 224.736 | - | 23.75 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 281.604 | - | 32.0352 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 48.3053 | - | 3.98438 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 79.9677 | - | 12.1953 | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 463.851 | - | 32.3008 | onnx_opset=13 |
| Classification | ResNet18 | - | - | - | - | - |  |
| Classification | ResNet34 | - | - | - | - | - |  |
| Classification | ResNet50 | - | - | - | - | - |  |
| Classification | ViT-tiny | (224, 224) | 3 | 307.119 | - | 12.3984 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 406.89 | - | 38.4805 | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 2811.66 | - | 275.512 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 939.852 | - | 46.3281 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 2771.37 | - | 88.0156 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 5675.15 | - | 154.891 | onnx_opset=13 |

## Raspberry Pi Zero 2 W

### FP16

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 1163.35 | - | 114.922 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 225.104 | - | 39.8477 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 315.645 | - | 63.7773 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 448.629 | - | 95.4648 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 32.5132 | - | 23.1523 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 94.6444 | - | 56.9414 | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 869.398 | - | 116.02 | onnx_opset=13 |
| Classification | ResNet18 | (224, 224) | 3 | 441.127 | - | 120.402 | onnx_opset=13 |
| Classification | ResNet34 | (224, 224) | 3 | 817.117 | - | 185.582 | onnx_opset=13 |
| Classification | ResNet50 | (224, 224) | 3 | 940.785 | - | 233.328 | onnx_opset=13 |
| Classification | ViT-tiny | (224, 224) | 3 | 713.305 | - | 58.3203 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 942.47 | - | 116.637 | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 5779.5 | - | 264.441 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 2515.49 | - | 148.547 | onnx_opset=13 |

### INT8

| Task | Model | Input shape | Classes | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) | Ramarks |
|---|---|---|---|---|---|---|---|
| Classification | EfficientFormer-l1 | (224, 224) | 3 | 248.033 | - | 23.6562 | onnx_opset=13 |
| Classification | MixNet-s | (224, 224) | 3 | 152.92 | - | 13.2109 | onnx_opset=13 |
| Classification | MixNet-m | (224, 224) | 3 | 257.34 | - | 20.4414 | onnx_opset=13 |
| Classification | MixNet-l | (224, 224) | 3 | 340.322 | - | 28.0742 | onnx_opset=13 |
| Classification | MobileNetV3-small | (224, 224) | 3 | 78.6897 | - | 5.73438 | onnx_opset=13 |
| Classification | MobileNetV3-large | (224, 224) | 3 | 132.282 | - | 13.7188 | onnx_opset=13 |
| Classification | MovileViT-s | (256, 256) | 3 | 701.015 | - | 30.9727 | onnx_opset=13 |
| Classification | ResNet18 | - | - | - | - | - |  |
| Classification | ResNet34 | - | - | - | - | - |  |
| Classification | ResNet50 | - | - | - | - | - |  |
| Classification | ViT-tiny | (224, 224) | 3 | 463.293 | - | 13.2422 | onnx_opset=13 |
| Segmentation | PIDNet-s | (512, 512) | 35 | 574.138 | - | 32.5078 | onnx_opset=13 |
| Segmentation | SegFormet-b0 | (512, 512) | 35 | 4373.37 | - | 148.633 | onnx_opset=13 |
| Detection | YOLOX-s | (640, 640) | 4 | 1337.21 | - | 40.3945 | onnx_opset=13 |
| Detection | YOLOX-m | (640, 640) | 4 | 3442.01 | - | 79.5859 | onnx_opset=13 |
| Detection | YOLOX-l | (640, 640) | 4 | 7061.52 | - | 143.648 | onnx_opset=13 |
