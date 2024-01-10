from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    seed: int = 1
    num_workers: int = 4
    gpus: str = "0"
