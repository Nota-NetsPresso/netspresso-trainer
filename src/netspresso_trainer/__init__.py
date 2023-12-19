from pathlib import Path

from netspresso_trainer.cfg import TrainerConfig
from netspresso_trainer.trainer_cli import parse_args_netspresso, train_cli
from netspresso_trainer.trainer_inline import export_config_as_yaml, train_with_config, train_with_yaml
from netspresso_trainer.trainer_util import validate_config

### Starting from v0.0.9, the default train function runs with yaml configuration
train = train_with_yaml
# train = train_with_config

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
