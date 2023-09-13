#!/bin/bash

cd "$(dirname ${0})/.."
python -m ruff check src/netspresso_trainer --fix