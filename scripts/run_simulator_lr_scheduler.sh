#!/bin/bash

cd "$(dirname ${0})/.."

python demo/gradio_lr_scheduler.py\
  --docs demo/docs/description_lr_scheduler.md\
  --config config/training/template/common.yaml\
  --local --port 50002