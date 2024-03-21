from abc import ABC, abstractmethod


class BasePipeline(ABC):
    def __init__(self):
        super(BasePipeline, self).__init__()