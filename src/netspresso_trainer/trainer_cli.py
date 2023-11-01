import argparse
import os
import subprocess
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf

from netspresso_trainer.trainer_common import train_common

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


def run_distributed_training_script(gpu_ids, data, augmentation, model, training, logging, environment, log_level):
    
    command = [
        "--data", data,
        "--augmentation", augmentation, 
        "--model", model,
        "--training", training,
        "--logging", logging,
        "--environment", environment,
        "--log_level", log_level,
    ]
    
    # Distributed training script
    command = [
        'python', '-m', 'torch.distributed.launch',
        f'--nproc_per_node={len(gpu_ids)}',  # GPU #
        os.path.abspath(__file__), *command
    ]

    # Run subprocess
    process = subprocess.Popen(command)

    try:
        process.wait()
    except KeyboardInterrupt:
        print("Interrupted. Terminating the training process...")
        process.terminate()
        process.wait()


def parse_gpu_ids(gpu_arg: str):
    """Parse comma-separated GPU IDs and return as a list of integers."""
    try:
        gpu_ids = [int(id) for id in gpu_arg.split(',')]
        
        if len(gpu_ids) == 1:  # Single GPU
            return gpu_ids[0]
        
        gpu_ids = sorted(gpu_ids)
        return gpu_ids
    except ValueError as e:
        raise argparse.ArgumentTypeError('Invalid GPU IDs. Please provide comma-separated integers.') from e


def parse_args_netspresso(with_gpus=False):

    parser = argparse.ArgumentParser(description="Parser for NetsPresso configuration")

    # -------- User arguments ----------------------------------------
    
    if with_gpus:
        parser.add_argument(
            '--gpus', type=parse_gpu_ids, default=0,
            dest='gpus',
            help='GPU device indices (comma-separated)')

    parser.add_argument(
        '--data', type=str, required=True,
        dest='data',
        help="Config for dataset information")

    parser.add_argument(
        '--augmentation', type=str, required=True,
        dest='augmentation',
        help="Config for data augmentation")

    parser.add_argument(
        '--model', type=str, required=True,
        dest='model',
        help="Config for the model architecture")

    parser.add_argument(
        '--training', type=str, required=True,
        dest='training',
        help="Config for training options")

    parser.add_argument(
        '--logging', type=str, default='config/logging.yaml',
        dest='logging',
        help="Config for logging options")

    parser.add_argument(
        '--environment', type=str, default='config/environment.yaml',
        dest='environment',
        help="Config for training environment (# workers, etc.)")

    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default=LOG_LEVEL,
        dest='log_level',
        help="Config for training environment (# workers, etc.)")

    args_parsed, _ = parser.parse_known_args()

    return args_parsed


def set_arguments(data: Union[Path, str], augmentation: Union[Path, str],
                  model: Union[Path, str], training: Union[Path, str],
                  logging: Union[Path, str], environment: Union[Path, str]) -> DictConfig:
    
    conf_data = OmegaConf.load(data)
    conf_augmentation = OmegaConf.load(augmentation)
    conf_model = OmegaConf.load(model)
    conf_training = OmegaConf.load(training)
    conf_logging = OmegaConf.load(logging)
    conf_environment = OmegaConf.load(environment)

    conf = OmegaConf.create()
    conf.merge_with(conf_data)
    conf.merge_with(conf_augmentation)
    conf.merge_with(conf_model)
    conf.merge_with(conf_training)
    conf.merge_with(conf_logging)
    conf.merge_with(conf_environment)
    
    return conf


def train_with_yaml_impl(gpus: Union[list, int], data: Union[Path, str], augmentation: Union[Path, str],
                         model: Union[Path, str], training: Union[Path, str],
                         logging: Union[Path, str], environment: Union[Path, str], log_level: str = LOG_LEVEL):
    
    assert isinstance(gpus, (list, int))
    gpu_ids_str = ','.join(map(str, gpus)) if isinstance(gpus, list) else str(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
    
    if isinstance(gpus, int):
        conf = set_arguments(data, augmentation, model, training, logging, environment)
        train_common(conf, log_level=log_level)
    else:
        run_distributed_training_script(gpus, data, augmentation, model, training, logging, environment, log_level)


def train_cli() -> None:
    args_parsed = parse_args_netspresso(with_gpus=True)
    
    train_with_yaml_impl(
        gpus=args_parsed.gpus,
        data=args_parsed.data,
        augmentation=args_parsed.augmentation,
        model=args_parsed.model,
        training=args_parsed.training,
        logging=args_parsed.logging,
        environment=args_parsed.environment,
        log_level=args_parsed.log_level
    )


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

    train_common(conf, log_level=args_parsed.log_level)


if __name__ == "__main__":
    
    # Execute by `run_distributed_training_script`
    train_cli_without_additional_gpu_check()