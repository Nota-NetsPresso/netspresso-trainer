import argparse

from omegaconf import OmegaConf

from .models.builder import build_model
from .pipelines import ClassificationPipeline, SegmentationPipeline
from .utils.environment import set_device

SUPPORT_TASK = ['classification', 'segmentation']

def parse_args_netspresso():

    parser = argparse.ArgumentParser(description="Parser for NetsPresso configuration")
    
    # -------- User arguments ----------------------------------------
    
    parser.add_argument(
        '--config', type=str, default='',
        dest='config', 
        help="Config path")
    
    parser.add_argument(
        '-o', '--output_dir', type=str, default='..',
        dest='output_dir', 
        help="Checkpoint and result saving directory")

    args, _ = parser.parse_known_args()    
    
    return args

def train():
    args_parsed = parse_args_netspresso()
    args = OmegaConf.load(args_parsed.config)
    distributed, world_size, rank, devices = set_device(args)
    
    args.distributed = distributed
    args.world_size = world_size
    args.rank = rank
    
    task = str(args.train.task).lower()
    assert task in SUPPORT_TASK
    
    model = build_model(args)
    
    if task == 'classification':
        trainer = ClassificationPipeline(args, model, devices)
    elif task == 'segmentation':
        trainer = SegmentationPipeline(args, model, devices)
    else:
        raise AssertionError(f"No such task! (task: {task})")
    
    trainer.train()