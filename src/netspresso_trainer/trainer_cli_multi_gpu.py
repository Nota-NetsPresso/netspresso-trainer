import os

from netspresso_trainer.trainer_common import train_common
from netspresso_trainer.trainer_util import parse_args_netspresso, set_arguments

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


def train_cli_without_additional_gpu_check() -> None:
    args_parsed = parse_args_netspresso(with_gpus=False)

    conf = set_arguments(
        data=args_parsed.data,
        augmentation=args_parsed.augmentation,
        model=args_parsed.model,
        training=args_parsed.training,
        logging=args_parsed.logging,
        environment=args_parsed.environment
    )

    train_common(
        conf,
        task=args_parsed.task,
        model_name=args_parsed.model_name,
        is_graphmodule_training=args_parsed.is_graphmodule_training,
        logging_dir=args_parsed.logging_dir,
        log_level=args_parsed.log_level
    )


if __name__ == "__main__":

    # Execute by `run_distributed_training_script`
    train_cli_without_additional_gpu_check()
