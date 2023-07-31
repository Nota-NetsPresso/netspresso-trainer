import os

from utils.logger import set_logger
from train_common import set_arguments, train

logger = set_logger('train', level=os.getenv('LOG_LEVEL', 'INFO'))

if __name__ == '__main__':
    args_parsed, args = set_arguments(is_graphmodule_training=True)
    train(args_parsed, args, is_graphmodule_training=True)