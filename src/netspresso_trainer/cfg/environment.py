from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    num_workers: int = 4
    gpus: str = "0"
