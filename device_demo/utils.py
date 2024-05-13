import time
from typing import Optional


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


class Timer:
    """Basic timer with CRUD functions
    """

    def __init__(self) -> None:
        self.history = {}

    def start_record(self, name) -> bool:
        if name not in self.history:
            self.history[name] = TimeRecode()  # create
            return True
        return False  # fail cause alreay exists

    def _end_record(self, name) -> bool:
        if name in self.history:
            self.history[name].end()  # update
            return True  # success
        return False  # fail cause no such key

    def end_record(self, name):
        self._end_record(name)

    def get(self, name, as_pop=True) -> Optional[float]:
        if name in self.history:
            record = self.history.pop(name) if as_pop else self.history[name]  # read (+ delete)
            if not record.done:
                record.end()
                assert record.done
            return record.elapsed

        return  # no such key

