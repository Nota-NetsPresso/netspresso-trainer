from typing import List, TypedDict


class ProcessorStepOut(TypedDict):
    images: List
    pred: List
    target: List

    @classmethod
    def empty(cls):
        return cls(images=[], pred=[], target=[])
