from typing import List, TypedDict


class ProcessorStepOut(TypedDict):
    name: List
    pred: List
    target: List

    @classmethod
    def empty(cls):
        return cls(name=[], pred=[], target=[])
