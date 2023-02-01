import os
from pipelines.base import BasePipeline
from utils.logger import set_logger

logger = set_logger('pipelines', level=os.getenv('LOG_LEVEL', default='INFO'))

class SegmentationPipeline(BasePipeline):
    pass