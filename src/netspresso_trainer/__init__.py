# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from pathlib import Path

from netspresso_trainer.evaluator_main import evaluation_cli
from netspresso_trainer.inferencer_main import inference_cli
from netspresso_trainer.trainer_main import parse_args_netspresso, train_cli, train_with_yaml

### Starting from v0.0.9, the default train function runs with yaml configuration
train = train_with_yaml
# train = train_with_config

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
