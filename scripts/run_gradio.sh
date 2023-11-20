#!/bin/bash

cd "$(dirname ${0})/.."

export PATH_AUG_DOCS="demo/docs/description_augmentation.md"
export PATH_SCHEDULER_DOCS="demo/docs/description_scheduler.md"
export PATH_PYNETSPRESSO_DOCS="demo/docs/description_pynetspresso.md"
export PATH_CONFIG_ROOT="config/"

python demo/app.py\
  --image assets/kyunghwan_cat.jpg\
  --local --port 50003