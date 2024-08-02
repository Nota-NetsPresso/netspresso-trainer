# YOLOX-s

We provide TFLite runtime code to run models on devices, which can be found in the `tools/device_runtime` directory. This document presents an example of deploying and running the object detection model, YOLOX-s.

## Install Miniconda (Optional)

Install Miniconda from Miniforge.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh # Agree and install
```

After initiate miniconda, create virtual environment for tflite runtime.

```
conda create -n tflite_runtime python=3.8.16
conda activate tflite_runtime
```

## Ready runtime code

To utilize provided TFLite runtime code, you should clone our repository into your device.

```bash
git clone -b master https://github.com/Nota-NetsPresso/netspresso-trainer.git
cd netspresso_trainer/tools/device_runtime
```

## Install packages

Install python packages with `requirements-tflite.txt`.

```bash
pip install -r requirements-tflite.txt
```

## Set runtime configuration

We have implemented the runtime code to accept a config file as input. The runtime configuration contains information about the model as well as details for preprocessing and postprocessing. For consistency, the structure of the preprocessing configuration aligns with the training augmentation configuration, and the postprocessor configuration structure aligns with the model configuration. Therefore, to run the YOLOX-s model with TFLite, you need to prepare the YAML configuration as following.

```yaml
runtime:
  task: detection
  model_path: ./yolox_s.tflite
  preprocess:
    - 
      name: resize
      size: 640
      interpolation: bilinear
      max_size: null
      resize_criteria: long
    - 
      name: pad
      size: 640
      fill: 114
  postprocess:
    score_thresh: 0.4
    nms_thresh: 0.65
```

| Field <img width=200/> | Description |
|---|---|
| `task` | (str) The type of task the model is designed for. We only support "detection" now. |
| `model_path` | (str) TFLite model tath to run. |
| `preprocess` | (list) The preprocessing pipeline to be applied to the input image. Refer to [Augmentation page](../../components/augmentation/overview.md) for more details. |
| `postprocess` | (dict) The postprocessing hyper parameters to be applied to the outputs. This field can be different according to the model. Refer to [Model page](../../components/model/overview.md) for more details. |

## Run with camera input

Execute the tflite_run.py Python file to run the YOLOX-s model on your device. Ensure that a camera is connected to the device, as the input images will be captured from the camera. When the file is executed, the process will read images from the connected camera, process them, and use the model to predict bounding boxes. These bounding boxes will be drawn using the OpenCV library and displayed in real-time in a window.

```bash
python tflite_run.py --config-path ./config/yolox-s-tflite.yaml
```

## Run with local dataset

To be updated ...