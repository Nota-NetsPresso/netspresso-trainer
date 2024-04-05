from pathlib import Path

from netspresso_trainer.evaluator_main import evaluation_cli
from netspresso_trainer.inferencer_main import inference_cli
from netspresso_trainer.trainer_main import parse_args_netspresso, train_cli, train_with_yaml

### Starting from v0.0.9, the default train function runs with yaml configuration
train = train_with_yaml
# train = train_with_config

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
