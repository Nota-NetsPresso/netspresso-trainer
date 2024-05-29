import time

class TimeRecode:
    def __init__(self) -> None:
        self._start = time.time()
        self._end = None

    def end(self) -> None:
        self._end = time.time()

    @property
    def done(self) -> bool:
        return self._end is not None

    @property
    def elapsed(self) -> float:
        return self._end - self._start if self._end is not None else None
