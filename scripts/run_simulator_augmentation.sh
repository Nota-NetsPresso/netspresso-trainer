#!/bin/bash

cd "$(dirname ${0})/.."

python demo/gradio_augmentation.py\
  --docs demo/docs/description_augmentation.md\
  --config config/augmentation/template/common.yaml\
  --image assets/kyunghwan_cat.jpg\
  --local --port 50003